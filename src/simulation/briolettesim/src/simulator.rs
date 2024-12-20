use absim::graph_utils::GraphVertexIndex;
use rand::prelude::*;
use rand::SeedableRng;
use std::cmp::{max, min};
use std::sync::{Arc, RwLock};

use absim::extras::SimulationPopulation;
use absim::{
    Address, Agent, Enqueue, Event, EventQueue, Population, Simulation,
    WorldView,
};

use crate::{AgentData, AgentRole, BankData, Coin, CoinState, EventData, GossipData, MerchantData, PopulationDel, RegisterData, RequestRole, RequestTransactionData, ResourceData, SimulationTools, SimulatorHelpers, Statistics, SyncState, SynchronizeData, TransactData, TransactionCoin, ValidateData, ValidateResponseData, ViewData, WorldData};


#[derive(Debug)]
pub struct Simulator {
    //pub rng: Arc<RwLock<StdRng>>, // Interior Mutability!
    pub seed: u64,
    // TODO: hide this in absim
    pub clones: Arc<RwLock<usize>>,            // Interior Mutability
    pub helper: Arc<RwLock<SimulatorHelpers>>, // Violate ro-ness with interior mutability
}
impl Simulator {
    pub fn new(seed: u64, helper: SimulatorHelpers) -> Self {
        Self {
            seed: seed,
            clones: Arc::new(RwLock::new(0)),
            helper: Arc::new(RwLock::new(helper)),
        }
    }
    // TODO add error handling
    pub fn do_transaction(
        &self,
        view: &ViewData,
        helper: &mut SimulatorHelpers,
        source: &Agent<AgentData>,
        target: &Agent<AgentData>,
        amount: usize,
        double_spend: bool,
        pop: bool,
        coin_count: &mut usize,
        queue: &mut impl Enqueue<Self>,
    ) -> usize {
        let mut stats = Statistics::default();
        let mut count = 0;
        let txn_epoch = max(source.data.epoch, target.data.epoch);
        // Gossip here to ensure we're in sync.
        if source.data.epoch != target.data.epoch {
            // println!("Source epoch: {}, target epoch: {}", source.data.epoch, target.data.epoch);
            queue.enqueue(
                Address::AgentId(source.id),
                Address::AgentId(target.id),
                EventData::Gossip(GossipData { epoch: txn_epoch }),
            );
            count += 1;
        }
        // We check for revocation of the peer based on the max() of their epochs.
        // While it is unlikely for a revoked party to share a revoked update, they shouldn't be
        // able to share a secure epoch time that doesn't include updates because the time and
        // hash/root hash must be linked.  If epochs don't need ordering, they can also just be
        // signatures over hashes, but we'd need to figure out how deltas apply then, etc.
        //
        // We also check if either is revoked now, pretended the receiving agent has vetted and
        // rejected.  This avoids doing rejections in the apply() phase.
        assert!(txn_epoch <= view.epoch);
        let mut rejected = false;
        if view.epochs[txn_epoch].revocation.contains(&source.id) {
            queue.enqueue(
                Address::AgentId(source.id),
                Address::NoAddress,
                EventData::RejectedTransact,
            );
            stats.txns_rejected_total += 1;
            rejected = true;
        } else if view.epochs[txn_epoch].revocation.contains(&target.id) {
            queue.enqueue(
                Address::NoAddress,
                Address::AgentId(target.id),
                EventData::RejectedTransact,
            );
            stats.txns_rejected_total += 1;
            rejected = true;
        } else {
            stats.txns_total += 1;
        }
        if source.data.role.is_consumer() && target.data.role.is_consumer() {
            stats.txns_p2p_rejected_total = stats.txns_rejected_total;
            stats.txns_p2p_total = stats.txns_total;
        } else if source.data.role.is_consumer() && target.data.role.is_merchant() {
            stats.txns_p2m_rejected_total = stats.txns_rejected_total;
            stats.txns_p2m_total = stats.txns_total;
        } else if source.data.role.is_merchant() && target.data.role.is_merchant() {
            stats.txns_m2m_rejected_total = stats.txns_rejected_total;
            stats.txns_m2m_total = stats.txns_total;
        } else if source.data.role.is_merchant() && target.data.role.is_bank() {
            stats.txns_m2b_rejected_total = stats.txns_rejected_total;
            stats.txns_m2b_total = stats.txns_total;
        } else if source.data.role.is_bank() && target.data.role.is_merchant() {
            stats.txns_b2m_rejected_total = stats.txns_rejected_total;
            stats.txns_b2m_total = stats.txns_total;
        } else if source.data.role.is_consumer() && target.data.role.is_bank() {
            stats.txns_p2b_rejected_total = stats.txns_rejected_total;
            stats.txns_p2b_total = stats.txns_total;
        } else if source.data.role.is_bank() && target.data.role.is_consumer() {
            stats.txns_b2p_rejected_total = stats.txns_rejected_total;
            stats.txns_b2p_total = stats.txns_total;
        } else if source.data.role.is_bank() && target.data.role.is_bank() {
            stats.txns_b2b_rejected_total = stats.txns_rejected_total;
            stats.txns_b2b_total = stats.txns_total;
        }

        if stats != Statistics::default() {
            queue.enqueue(
                Address::AgentId(source.id),
                Address::World,
                EventData::UpdateStatistics(stats),
            );
        }
        count += 1;

        // Don't process the txn if it is rejected by either party based on revocation.
        if rejected {
            return count + 1;
        }

        let mut coins = Vec::new();
        // Build the payload
        let mut coin_iter = source.data.coins.iter().rev(); // Work backwards to enable popping!
        for _c in 0..amount {
            if let Some(coin) = coin_iter.next() {
                coins.push(TransactionCoin {
                    coin: coin.clone(),
                    copy: double_spend,
                    popped: pop,
                });
                *coin_count += 1;
                // Assign the recipient.
                coins
                    .iter_mut()
                    .last()
                    .unwrap()
                    .coin
                    .history
                    .push(target.id);
                coins
                    .iter_mut()
                    .last()
                    .unwrap()
                    .coin
                    .step_history
                    .push(view.step);
                // Create a unique txn id
                coins
                    .iter_mut()
                    .last()
                    .unwrap()
                    .coin
                    .tx_history
                    .push(helper.get_uniform(10, 9223372036854775808));
            }
        }
        queue.enqueue(
            Address::AgentId(source.id),
            Address::AgentId(target.id),
            EventData::Transact(TransactData { coins: coins }),
        );
        count + 1
    }
}
// Create a stable way to maintained seeded randomness across view and client splits.
impl Clone for Simulator {
    fn clone(&self) -> Self {
        let mut c = self.clones.write().unwrap();
        // Let's avoid silliness.
        if *c == usize::MAX {
            *c = 0;
        } else {
            *c += 1;
        }
        let m: u64 = (*c).try_into().unwrap();
        let mut new_help = self.helper.read().unwrap().clone();
        new_help.rng = Box::new(SeedableRng::seed_from_u64(self.seed + m));
        Self {
            seed: self.seed,
            clones: Arc::new(RwLock::new(*c)),
            helper: Arc::new(RwLock::new(new_help)),
        }
    }
}

impl Simulation for Simulator {
    type Data = AgentData;
    type Event = EventData;
    type World = WorldData;
    type View = ViewData;
    type SimulationPopulation = SimulationPopulation<Self>;
    type ViewPopulation = SimulationPopulation<Self>;

    fn worldview(
        &self,
        id: usize,
        population: &Self::SimulationPopulation,
        world: &Self::World
    ) -> Option<WorldView<Self>> {
        if id >= world.graph.vertices.len() {
            return None;
        }
        let location = world.graph.get_location(id);
        let cell = world.graph.at_location(&GraphVertexIndex(location.clone()));
        /*
        println!(
            "Generating cellular World View: ({}, {})",
            location.0, location.1
        );
        */
        // Create our own population copy.
        let mut pop = Box::new(SimulationPopulation::new());
        for id in &cell.agents {
            // Keep same ids
            pop.update(&population.get(id.clone()).unwrap().clone());
        }
        let mut resource = ResourceData::default();
        if cell.resources.len() > 0 {
            resource = world.resources[cell.resources[0]].clone();
        }
        // TODO: Don't copy all epochs over time, but still more efficient than putting it in each
        // agent.
        Some(WorldView::new(
            pop,
            ViewData {
                id: id,
                step: world.step,
                bounds: world.bounds,
                resource: resource,
                epoch: world.epoch,
                epochs: world.epochs.clone(),
                graph: world.graph.clone()
            }
        ))
    }

    fn generate(
        &self,
        agent: &Agent<Self::Data>,
        view: &WorldView<Self>,
        queue: &mut EventQueue<Self>
    ) -> usize {
        let mut count = 0;
        let helper = &mut *self.helper.write().expect("interior mutability");
        let stats = Statistics::default();

        for e in &agent.data.pending {
            queue.enqueue(e.source, e.target, e.data.clone());
            count += 1;
        }
        queue.enqueue(
            Address::NoAddress,
            Address::AgentId(agent.id),
            EventData::Age(view.data().step),
        );
        count += 1;

        match &agent.data.role {
            // Generate bank events
            AgentRole::Bank(_data) => {
                // Syncs every tick even if it doesnt validate every tick.
                queue.enqueue(
                    Address::AgentId(agent.id),
                    Address::World,
                    EventData::Synchronize(SynchronizeData {}),
                );
                count += 1;
            }
            //
            AgentRole::Merchant(data) => {
                if let Some(_tx_step) = data.last_tx_step {
                    // TODO: Make deposit timeframe configurable
                    // Deposit 4 times a day.
                    if view.data().step % 8 == 0 {
                        // Construct a dummy bank so we can reuse do_transaction()
                        let mut bank = Agent {
                            id: data.bank,
                            data: AgentData::default(),
                        };
                        // Bank is always up-to-date.
                        // For other "online" we'll need to RequestInfo to get the recipient's epoch. Or enforce a Sync before online.
                        bank.data.epoch = view.data().epoch;
                        bank.data.role = AgentRole::Bank(BankData::default());
                        // Currently, amount is just # of coins.
                        // let amount = agent.data.coins.iter().map(|c| c.value).sum();
                        let amount = agent.data.coins.len();
                        let mut coin_count = 0;
                        // TODO: Enable deposit double spend attempts too, etc.
                        if amount > 0 {
                            count += self.do_transaction(
                                &view.data(),
                                helper,
                                agent,
                                &bank,
                                amount,
                                false,
                                false,
                                &mut coin_count,
                                queue,
                            );
                        }
                    }
                }
            }
            // Generate consumer events
            AgentRole::Consumer(data) => {
                // Required before any other work.
                if agent.data.registered == false {
                    // Later enables hwids to be added, etc and double spenders to be filtered?
                    queue.enqueue(
                        Address::AgentId(agent.id),
                        Address::World,
                        EventData::Register(RegisterData {}),
                    );
                    count += 1;
                    return count;
                }
                if helper.probability_check(&data.sync_distribution, data.sync_probability)
                    || data.wids < data.wid_low_watermark
                {
                    // Later enables hwids to be added, etc.
                    queue.enqueue(
                        Address::AgentId(agent.id),
                        Address::World,
                        EventData::Synchronize(SynchronizeData {}),
                    );
                    count += 1;
                }
                // TODO Add low threshold for withdrawl and other models like withdraw-on-demand!
                if agent.data.coins.len() < 2
                    && data.account_balance > 5
                    && view.data().step > data.last_requested_step + 1
                {
                    // TODO: We should enqueue a Gossip here but we'll let the bank fire it off.
                    queue.enqueue(
                        Address::AgentId(agent.id),
                        Address::AgentId(data.bank),
                        EventData::RequestTransaction(RequestTransactionData {
                            amount: min(data.account_balance, 5),
                            epoch: agent.data.epoch,
                            role: RequestRole::Consumer,
                        }),
                    );
                    // println!("[Agent {}] My bank is {} and my coins are {} and my balance is {}", agent.id, data.bank, agent.data.coins.len(), data.account_balance);
                    count += 1;
                }
                // If the consumer can and wants to transact.
                // TODO: Add support for _multiple_ transaction per-step, like at a hawker market
                // Add support for peer and merchant in same step.
                // E.g., avail_wids = wids; avail_coins = coins.len();
                // while cnt < per_period && avail_wids > 0 && avail_coins > 0 ...
                if agent.data.registered && agent.data.coins.len() > 0 && data.wids > 0 {
                    let mut peer = None;
                    if helper.probability_check(&data.p2m_distribution, data.p2m_probability) {
                        let merchant_iter = view.population().agents.iter().filter(|entry| {
                            match entry.1.data.role {
                                AgentRole::Merchant(_) => true,
                                _ => false,
                            }
                        });
                        if let Some(merchant_entry) = merchant_iter.choose(&mut helper.rng) {
                            // println!("Picked a merchant to transact with! {}", merchant_entry.0);
                            peer = view.population().agents.get(merchant_entry.0);
                        }
                    } else if helper.probability_check(&data.p2p_distribution, data.p2p_probability)
                    {
                        let peer_iter =
                            view.population().agents.iter().filter(|entry| {
                                match entry.1.data.role {
                                    AgentRole::Consumer(_) => true,
                                    _ => false,
                                }
                            });
                        if let Some(peer_entry) = peer_iter.choose(&mut helper.rng) {
                            // println!("Picked a peer to transact with! {}", peer_entry.0);
                            peer = view.population().agents.get(peer_entry.0);
                        }
                    }
                    if let Some(receiver) = peer {
                        // Will we double spend?
                        let ds = data.double_spend_probability > 0.0
                            && helper.probability_check(
                                &data.double_spend_distribution,
                                data.double_spend_probability,
                            );
                        // Gossip & transact
                        // TODO: Add transaction count enforcement per step period.
                        // Note: rng doesn't like it is the range is 0 (e.g., 1, 1).
                        // Also, if multiple txns are allowed per-step, the max can't be the coin
                        // list.
                        let amount = max(1, helper.rng.gen_range(0..agent.data.coins.len()));
                        let mut coin_count = 0;
                        count += self.do_transaction(
                            &view.data(),
                            helper,
                            agent,
                            receiver,
                            amount,
                            ds,
                            false,
                            &mut coin_count,
                            queue,
                        );
                    }
                }
                if helper.probability_check(&data.move_distribution, data.move_probability) {
                    let new_location =
                        helper.relocate(
                            &view.data().graph,
                            &data.move_distribution, 
                            &agent.data.location
                        );
                    // TODO: Figure out if we can get rid of the crawl without incurring worse overhead.
                    queue.enqueue(
                        Address::AgentId(agent.id),
                        Address::NoAddress,
                        //Address::View(view.id), // Tells the world what cell we were in without looking it up.
                        EventData::Move(new_location),
                    );
                    count += 1;
                }
            }
        }
        if stats != Statistics::default() {
            queue.enqueue(
                Address::AgentId(agent.id),
                Address::World,
                EventData::UpdateStatistics(stats),
            );
        }
        count + 1
    }

    fn apply(
        &self,
        agent: &mut Agent<Self::Data>,
        world: &Self::World,
        event: &Event<Self::Event>,
    ) {
        let helper = &mut *self.helper.write().expect("interior mutability");
        let mut stats = Statistics::default();
        // Since generate() cannot delete the pending events, clear the queue if it has old events
        // in it.
        if let Some(e) = agent.data.pending.iter().last() {
            if world.step > e.id {
                agent.data.pending.clear();
            }
        }
        // TODO add log levels
        // println!("agent {}: {:?}", agent.id, event);

        // Handle common event paths
        match &event.data {
            EventData::Gossip(gdata) => {
                // TODO: figure out how to count how long it took to reach world.epoch.
                let old = agent.data.epoch;
                agent.data.epoch = max(gdata.epoch, agent.data.epoch);
                // println!("New Epoch: {}", agent.data.epoch);
                
                // Double spenders don't actually update.
                if let AgentRole::Consumer(cdata) = &agent.data.role {
                    if cdata.double_spend_probability != 0.0 {
                        agent.data.epoch = old;
                    }
                }
            }
            EventData::Synchronize(_) => {
                if agent.data.epoch < world.epoch {
                    
                    // println!("New Epoch by synchronization: {}", world.epoch);
                    agent.data.epoch = world.epoch;
                }
            }
            EventData::Transact(txn) => {
                // If money is sent to an agent, they should:
                // 1. Check if the recipient is revoked and reject it (done in generate()!)
                // 2. Optionally queue up a Validate() call depending on their needs (TODO: see below)
                // 3. Add the coins to their coin list! (done)
                if event.target == Address::AgentId(agent.id) {
                    for tcoin in &txn.coins {
                        // Not common, but banks put money in holding first.
                        let mut coin = tcoin.coin.clone();
                        // The simulation knows this is a copy even if the operator doesn't yet.
                        if tcoin.copy {
                            coin.copied = true;
                            // Increment counterfeits total
                            stats.coins_double_spent_total += 1;
                            // Increment total coins in circulation, including counterfeits.
                            stats.coins_total += 1;
                        }
                        if let AgentRole::Bank(bank) = &mut agent.data.role {
                            bank.holding.push(coin);
                        } else {
                            agent.data.coins.push(coin);
                        }
                    }
                }
                // If money is sent by an agent, they must only remove it if it wasn't double spent.
                if event.source == Address::AgentId(agent.id) {
                    // For every coin we sent, remove it from our wallet.
                    // Assume the txn coins will never be _huge_ vs a potential bank coin list.
                    let mut removals = txn
                        .coins
                        .iter()
                        .filter_map(|c| {
                            if c.copy == false && c.popped == false {
                                Some(c.coin.clone())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<Coin>>();
                    // Keep track so we can stop early since we always takes from the tail.
                    let mut index = agent.data.coins.len();
                    while removals.len() > 0 {
                        index -= 1;
                        let coin = &agent.data.coins[index];
                        // Make sure we remove the exact match since double spending may result in
                        // duplicate IDs. In that case, we remove an unseen coin and make this agent look
                        // like a double spender.
                        if let Some(r) = removals.iter().position(|c| {
                            coin.id == c.id
                                && coin.tx_history[..(c.tx_history.len() - 1)]
                                    == c.tx_history[..(c.tx_history.len() - 1)]
                        }) {
                            agent.data.coins.swap_remove(index);
                            removals.swap_remove(r);
                        }
                        // We can always decrement because swap_remove() will pull the last element, which we will have already processed.
                        if index == 0 {
                            break;
                        }
                    }
                    assert!(removals.len() == 0);
                }
            }
            _ => {}
        }

        match &mut agent.data.role {
            AgentRole::Bank(data) => {
                match &event.data {
                    EventData::ValidateResponse(resp_data) => {
                        if event.target == Address::AgentId(agent.id) {
                            // Move ok coins out (_once_) and destroy bad coins.
                            // TODO: Request new coins from operator to replace bad coins.
                            let mut lists = resp_data.clone();
                            let mut index = 0;
                            while index < data.holding.len() {
                                let id = data.holding[index].id;
                                if let Some(position) =
                                    lists.counterfeit.iter().position(|c| *c == id)
                                {
                                    data.holding.swap_remove(index);
                                    lists.counterfeit.swap_remove(position);
                                    println!(
                                        "[agent {}] bank destroying bad coin {}",
                                        agent.id, id
                                    );
                                } else if let Some(position) =
                                    lists.ok.iter().position(|c| *c == id)
                                {
                                    agent.data.coins.push(data.holding.swap_remove(index));
                                    lists.ok.swap_remove(position);
                                } else {
                                    index += 1;
                                }
                            }
                        }
                    }
                    EventData::RequestTransaction(rt_data) => {
                        // A consumer is requesting funds which they should only request if they have
                        // the balance... but balance and coins may not match.
                        if event.target == Address::AgentId(agent.id) {
                            if let Address::AgentId(source) = event.source {
                                // Construct a dummy agent so we can reuse do_transaction()
                                let mut target_data: AgentData = AgentData::default();
                                match rt_data.role {
                                    RequestRole::Merchant => {
                                        target_data.role =
                                            AgentRole::Merchant(MerchantData::default())
                                    }
                                    RequestRole::Bank => {
                                        target_data.role = AgentRole::Bank(BankData::default())
                                    }
                                    _ => {} // Consumer is the default.
                                }
                                target_data.epoch = rt_data.epoch;
                                let target = Agent {
                                    id: source,
                                    data: target_data,
                                };
                                // TODO: Should apply() also just get a pre-fab ViewData that matches the agent's ViewData on generate()?
                                let view_data = ViewData {
                                    id: 0,
                                    step: world.step,
                                    bounds: world.bounds,
                                    resource: ResourceData::default(),
                                    epoch: world.epoch,
                                    epochs: world.epochs.clone(),
                                    graph: world.graph.clone()
                                };
                                let mut pending = Vec::new();
                                // WE must delete the coins or mark them as spent otherwise we'll double
                                // spend and search oru whole list for coins we don't have.
                                let mut pop_count = 0;
                                self.do_transaction(
                                    &view_data,
                                    helper,
                                    agent,
                                    &target,
                                    rt_data.amount,
                                    false,
                                    true,
                                    &mut pop_count,
                                    &mut pending,
                                );
                                agent.data.pending.append(&mut pending);
                                for _p in 0..pop_count {
                                    agent.data.coins.pop();
                                }
                                // Tag the last event so we don't drain this on a repeat call.
                                // Hacky but why waste yet another variable... :)
                                if let Some(e) = agent.data.pending.iter_mut().last() {
                                    e.id = world.step;
                                }
                            }
                        }
                    }
                    EventData::Transact(txn) => {
                        // The common code above should do all normal checks.
                        // Here we should enqueue a Validate(), etc.
                        // If we're the source, the coins were removed when we created the transaction
                        // above -- so there's no work.
                        // If we're the destination, add the coins to our reserves.
                        if event.target == Address::AgentId(agent.id) {
                            let mut new_coins = Vec::new();
                            for tcoin in &txn.coins {
                                new_coins.push(tcoin.coin.clone());
                            }
                            // We dont want to validate every coin in the bank every time, so queue up a
                            // validate for any deposit.
                            let validate_data = ValidateData { coins: new_coins };
                            agent.data.pending.push(Event {
                                id: world.step,
                                source: Address::AgentId(agent.id),
                                target: Address::World,
                                data: EventData::Validate(validate_data),
                            });
                        }
                    }
                    _ => {}
                }
            }
            AgentRole::Consumer(data) => {
                match &event.data {
                    EventData::Move(loc) => {
                        // Only move if we're the source
                        if let Address::AgentId(id) = event.source {
                            if id == agent.id {
                                agent.data.location = loc.clone();
                            }
                        }
                    }
                    EventData::Register(_) => {
                        // TODO: Add revoked-logic here since a double spender will have exposed
                        // their HWID which makes their device revocable permanently -- stopping them
                        // from re-registering.
                        agent.data.registered = true;
                    }
                    EventData::Synchronize(_) => {
                        // WIDs are issued blindly but it is known how many have been issued to a
                        // given linkable-proof/signature that is not linkable in other contexts.
                        // This enables WID issuance to be total limited per time. Here we just only
                        // ever top up to 8, but we shoud move to tracking time quanta totals per
                        // linked-sig.
                        // TODO: Discuss this more deeply.
                        // TODO: Swap with a GrantTickets event if world mutation is needed.
                        if data.wids < 8 {
                            // TODO: Operator determined.
                            data.wids = 8;
                        }
                    }
                    EventData::RequestTransaction(rt_data) => {
                        // If we withdrew money, decrement our balance
                        // and be patient for a step.
                        if event.source == Address::AgentId(agent.id)
                            && event.target == Address::AgentId(data.bank)
                        {
                            data.account_balance -= rt_data.amount;
                            data.last_requested_step = world.step;
                        }
                    }
                    EventData::Transact(_txn) => {
                        if event.source == Address::AgentId(agent.id) {
                            // Now decrement the WID unless the agent is a double spender
                            if stats.coins_double_spent_total == 0 {
                                // this event has not double spent.
                                data.wids -= 1;
                            }
                        }
                        // TODO: add support for deposit if over a wallet threshold.
                    }
                    EventData::RejectedTransact => {
                        if data.max_rejections > 0 {
                            data.max_rejections -= 1;
                        }
                        if data.max_rejections == 0 {
                            agent.data.pending.push(Event {
                                id: world.step,
                                source: Address::AgentId(agent.id),
                                target: Address::Population,
                                data: EventData::Depart(PopulationDel {
                                    ids: vec![agent.id],
                                }),
                            });
                        }
                    }
                    EventData::Age(_call_step) => {
                        data.lifetime -= 1;
                        if data.lifetime == 0 {
                            agent.data.pending.push(Event {
                                id: world.step,
                                source: Address::AgentId(agent.id),
                                target: Address::Population,
                                data: EventData::Depart(PopulationDel {
                                    ids: vec![agent.id],
                                }),
                            });
                        }
                    }
                    _ => {}
                }
            }
            AgentRole::Merchant(data) => match &event.data {
                EventData::Transact(_txn) => {
                    if event.target == Address::AgentId(agent.id) {
                        data.last_tx_step = Some(world.step);
                    }
                }
                EventData::Age(_call_step) => {
                    data.lifetime -= 1;
                    if data.lifetime == 0 {
                        agent.data.pending.push(Event {
                            id: world.step,
                            source: Address::AgentId(agent.id),
                            target: Address::Population,
                            data: EventData::Depart(PopulationDel {
                                ids: vec![agent.id],
                            }),
                        });
                    }
                }
                _ => {}
            },
        }
        if stats != Statistics::default() {
            agent.data.pending.push(Event {
                id: world.step,
                source: Address::AgentId(agent.id),
                target: Address::World,
                data: EventData::UpdateStatistics(stats),
            });
        }
    }

    // Perform distribution/consumption events from the current cell resources.
    fn view_generate(&self, _view: &WorldView<Self>, _queue: &mut EventQueue<Self>) -> usize {
        0
    }

    fn world_generate(
        &self,
        _population: &Self::SimulationPopulation,
        world: &Self::World,
        queue: &mut EventQueue<Self>,
    ) -> usize {
        // enqueue GlobalTick to Address::All if we want to allow singular event generation.
        /*
        queue.enqueue(
                    Address::World,
                    Address::NoAddress,
                    EventData::World(Tick),
                );
        */
        // TODO: Every n events, mint money
        // TODO: Every n events, update bank account balances.
        let mut count = 0;
        for e in &world.pending {
            queue.enqueue(e.source, e.target, e.data.clone());
            count += 1;
        }
        count
    }

    fn world_apply(
        &self,
        world: &mut Self::World,
        population: &Self::SimulationPopulation,
        events: &Vec<Event<Self::Event>>,
    ) {
        // Unlike agents, world_apply is called once per-tick.
        world.step += 1;
        world.pending.clear();

        // Update the grid here
        // First agents, then resources.
        //println!("Applying world events: {:?}", events);
        world.graph.reset();
        // Update world view
        for agent in population.agents.values() {
            world
                .graph
                .at_location_mut(&mut agent.data.location.clone())
                .agents
                .push(agent.id);
        }
        for r in 0..world.resources.len() {
            let resource = &world.resources[r];
            world
                .graph
                .at_location_mut(&mut resource.location.clone())
                .resources
                .push(r); // index == id
        }

        // Handle operator events now.
        for event in events {
            match &event.data {
                EventData::UpdateEpoch(_ue_data) => todo!(),
                EventData::Register(_) => {
                    world.statistics.register_total += 1;
                }
                EventData::Synchronize(_) => {
                    world.statistics.synchronize_total += 1;
                }
                EventData::UpdateStatistics(stats) => {
                    world.statistics.update(&stats);
                }
                EventData::Validate(vdata) => {
                    //println!("Validate: {:?}", event);
                    world.statistics.validate_total += 1;
                    let mut bad_coins = vec![];
                    let mut good_coins = vec![];
                    for coin in &vdata.coins {
                        if world.coin_map.contains_key(&coin.id) {
                            let known_coin_state = world.coin_map.get_mut(&coin.id).unwrap();
                            // If we already know this coin is bad, just collect the stats and move on.
                            if let Some(step) = known_coin_state.revoked {
                                bad_coins.push(coin.id);
                                // For this, we have to compute the different between the fork point and the history.
                                if let Some(txn_fork) = known_coin_state.fork_txn {
                                    // If a double spent coin is validated against the last known good history, then it shows up here because its
                                    // been revoked even though it will not have a txn_fork entry to consume.  We will push a 0 for that instead since
                                    // that is the reserved tx_history for minting.
                                    let coin_tx;
                                    if coin.tx_history.len() > txn_fork + 1 {
                                        coin_tx = coin.tx_history[txn_fork + 1];
                                        world.statistics.double_spent_most_txns = max(
                                            world.statistics.double_spent_most_txns,
                                            coin.history.len() - txn_fork,
                                        );
                                    } else {
                                        // To show we've now seen an untransferred bad coin.
                                        // TODO: Should we track if the double spender themselves checks in? Later, yes.
                                        // -> This might be how malware abused devices can be shown recoverable.
                                        // -> Need a innocent double spender profile where all their money is transferred away and they only spend bad coins.
                                        // -> Then see if we can catch the actual abuser because they will have received coins from the caught ds. (No HWID, so it'd be trace based.)
                                        // TODO: Add a system trace based on coin/coins to see how it looks as money moves.
                                        coin_tx = 0;
                                    }
                                    // Ignore already seen coins.
                                    if known_coin_state.forks.contains(&coin_tx) {
                                        // We've already seen this double spend path.
                                        // TODO: later note when the same bad coin is resubmitted as it shows the validator didn't delete it.
                                        world.statistics.coins_double_spent_recovered_repeats += 1;
                                        // Skip since we've seen it before.
                                        continue;
                                    } else {
                                        world.statistics.coins_double_spent_recovered += 1;
                                        known_coin_state.forks.push(coin_tx);
                                    }

                                    assert!(txn_fork <= coin.history.len());
                                    println!(
                                    "[operator] recovered double spend of coin {} (forked by Agent {}) after {} txns and {} steps",
                                    coin.id, coin.history[txn_fork], coin.history.len() - txn_fork, world.step - step
                                );
                                }
                                // in steps, not transfers. Transfers we can get from the history.
                                world.statistics.double_spent_longest_life = max(
                                    world.statistics.double_spent_longest_life,
                                    world.step - step,
                                );

                                continue;
                            }
                            let known_coin = &known_coin_state.coin;
                            // Now we see if the history is shorter or forked.
                            // If the coin coming in has a shorter history, it's a fork.
                            // If the history matches, then we can update.
                            let mut ds = None;
                            let mut ds_entry = None;
                            if coin.history.len() < known_coin.history.len() {
                                assert!(coin.history[0] == known_coin.history[0]);
                                assert!(coin.history.len() == coin.tx_history.len());
                                for entry in 1..coin.history.len() {
                                    if coin.history[entry] != known_coin.history[entry]
                                        || coin.tx_history[entry] != known_coin.tx_history[entry]
                                    {
                                        ds = Some(coin.history[entry - 1]);
                                        ds_entry = Some(entry - 1);
                                        break;
                                    }
                                }
                            } else {
                                assert!(coin.history[0] == known_coin.history[0]);
                                assert!(coin.history.len() == coin.tx_history.len());
                                for entry in 1..known_coin.history.len() {
                                    if coin.history[entry] != known_coin.history[entry]
                                        || coin.tx_history[entry] != known_coin.tx_history[entry]
                                    {
                                        ds = Some(coin.history[entry - 1]);
                                        ds_entry = Some(entry - 1);
                                        break;
                                    }
                                }
                            }
                            if let Some(ds_agent) = ds {
                                bad_coins.push(coin.id);
                                // Revoke the coin in our state map.
                                if let Some(entry) = ds_entry {
                                    known_coin_state.revoked = Some(coin.step_history[entry]);
                                    known_coin_state.fork_txn = Some(entry);
                                    known_coin_state.forks.push(coin.tx_history[entry + 1]);
                                }

                                // Create revocation sync data and update global state.
                                if world.epochs[world.epoch].revocation.contains(&ds_agent) == false
                                {
                                    println!(
                                    "[operator] double spending by agent {} detected. Revoking . . .",
                                    ds_agent
                                );
                                    println!("Coin: {:?}", coin);
                                    println!("Known coin: {:?}", known_coin);
                                    let last_step = world.epochs[world.epoch].step;
                                    let last_revocation =
                                        world.epochs[world.epoch].revocation.clone();
                                    // Add to an existing SyncState if the step hasn't changed.
                                    if last_step != world.step {
                                        world.epoch += 1;
                                        // TODO: Add revocation where everyone in a group is quarantined and cannot
                                        //       transact without a fresh registration and then filter the double spender
                                        //       out for human handling since we learned their hwid during DS.  This should allow
                                        //       device recovery and not depend on wid expiry to DS enforcement.
                                        //       Essentially "group" revocation.
                                        world.epochs.push(SyncState {
                                            id: world.epoch,
                                            revocation: last_revocation,
                                            quarantine: Vec::new(),
                                            step: world.step,
                                        });
                                    }
                                    world.epochs[world.epoch].revocation.push(ds_agent);

                                    world.statistics.double_spenders_revoked_total += 1;
                                }

                                // Validate doesnt happen on Sync.
                                // In Sync, we can't recover or see coins unless that is
                                // _required_. E.g., show what you have to get more wids? That
                                // seems risky for tracking.
                                world.statistics.coins_double_spent_recovered += 1;
                                // Now remove the coin from the Updater
                                // TODO: Add counter to hash map.
                                let survived_txns = coin.history.len() - ds_entry.unwrap();
                                // Start from the first transfer
                                let survived_steps =
                                    world.step - coin.step_history[ds_entry.unwrap() + 1];
                                println!(
                                "[operator] recovered double spend of coin {} after {} txns and {} steps",
                                coin.id, survived_txns, survived_steps,
                            );
                                /*
                                    println!("New {:?}", coin.history);
                                    println!("New {:?}", coin.tx_history);
                                    println!("Known {:?}", known_coin.history);
                                    println!("Known {:?}", known_coin.tx_history);
                                */
                                // Don't insert the bogus history.
                                continue;
                            } else {
                                // Often, the first double spend creates the first legitimate history entry.
                                // if coin.copied {
                                //  println!("We didn't detect foul play with a copied coin. First entry? {:?}", coin);
                                //}
                            }
                        }
                        // Update the history if the coin is good.
                        world.coin_map.insert(coin.id, CoinState::new(coin.clone()));
                        good_coins.push(coin.id);
                    }
                    let resp_data = ValidateResponseData {
                        ok: good_coins,
                        counterfeit: bad_coins,
                    };
                    world.pending.push(Event {
                        id: world.step,
                        source: Address::World,
                        target: event.source,
                        data: EventData::ValidateResponse(resp_data),
                    });
                }
                EventData::ValidateResponse(_resp_data) => {}
                _ => todo!(),
            }
        }
    }
    fn population_apply(
        &self,
        population: &mut Self::SimulationPopulation,
        _world: &Self::World,
        events: &Vec<Event<Self::Event>>,
    ) {
        // TODO log levels
        // println!("population: {:?}", events);
        for event in events {
            match &event.data {
                EventData::Arrive(add_data) => {
                    // TODO: loglevels println!("Adding {} agents and resources", add_data.count);
                    population.new_agents(&add_data.data, add_data.count);
                }
                // TODO clean up Depart
                EventData::Depart(del_data) => {
                    for id in &del_data.ids {
                        println!("[agent {}] is departing", id);
                        population.remove(id);
                    }
                }
                _ => {}
            }
        }
    }
}


impl Enqueue<Simulator> for Vec<Event<EventData>> {
    fn enqueue(&mut self, source: Address<usize>, target: Address<usize>, data: EventData) {
        self.push(Event {
            id: self.len(),
            source,
            target,
            data,
        });
    }
}
