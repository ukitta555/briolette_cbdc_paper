// Copyright 2023 The Briolette Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//  Basic model describing the briolette digital currency system.

pub mod simulator;

use absim::graph_utils::{GraphVertexIndex, SimulationGraph};
use rand::prelude::*;
use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use simulator::Simulator;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::{io, process};

use absim::clients::LocalSimulationClient;
use absim::extras::SimulationPopulation;
use absim::{
    Address, Event, Manager, ManagerInterface,
};
use levy_distr::Levy;
use rand_distr::{Pareto, Uniform};
use rand_flight::Flight;


#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct Coin {
    id: usize, // Since our 0 history isn't unique
    value: usize,
    copied: bool, // True when a copy was made. This lets us track circulation of ds.
    history: Vec<usize>, // agent index. Later we will need group for quarantine
    tx_history: Vec<usize>, // this lets us add ephemerality to catch respend of the same coin to
    // the same agent A(1) -> A(2) will make the same provenance
    step_history: Vec<usize>, // tracks whcih step the tx took place in
}

// Used in the coin map to track counterfeiting impact
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
    struct CoinState {
        coin: Coin,
        revoked: Option<usize>, // Step of revocation
        fork_txn: Option<usize>,
        forks: Vec<usize>, // List of known forks so we can accurately coin recoveries.
    }

impl CoinState {
    pub fn new(coin: Coin) -> Self {
        CoinState {
            coin: coin,
            revoked: None,
            fork_txn: None,
            forks: vec![],
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct SyncState {
    id: usize,
    revocation: Vec<usize>, // list of agent ids for now.
    quarantine: Vec<usize>, // group ids to manage WID issuance?
    step: usize,            // Create at what step
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct TransactionCoin {
    coin: Coin,
    copy: bool,   // true to double spend
    popped: bool, // True if the coin has already been removed from the sender.
}
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct TransactData {
    coins: Vec<TransactionCoin>, // double spending is literal this way.
}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
pub struct Statistics {
    potential_double_spender_max: usize,
    double_spenders_total: usize,
    double_spenders_revoked_total: usize,
    txns_total: usize,
    txns_p2p_total: usize,
    txns_p2m_total: usize,
    txns_m2m_total: usize,
    txns_b2m_total: usize,
    txns_m2b_total: usize,
    txns_b2p_total: usize,
    txns_p2b_total: usize,
    txns_b2b_total: usize,
    txns_p2p_rejected_total: usize,
    txns_p2m_rejected_total: usize,
    txns_m2m_rejected_total: usize,
    txns_b2m_rejected_total: usize,
    txns_m2b_rejected_total: usize,
    txns_b2p_rejected_total: usize,
    txns_p2b_rejected_total: usize,
    txns_b2b_rejected_total: usize,
    txns_rejected_total: usize,
    register_total: usize,
    synchronize_total: usize,
    coins_total: usize,
    coins_valid_total: usize,
    coins_double_spent_recovered: usize,
    coins_double_spent_recovered_repeats: usize,
    coins_double_spent_total: usize,
    double_spent_longest_life: usize,
    double_spent_most_txns: usize,
    validate_total: usize,
    #[serde(skip_serializing)]
    double_spent_life_measurements: Vec<usize>,
    #[serde(skip_serializing)]
    double_spent_txs_measurements: Vec<usize>
}

impl Statistics {
    pub fn exit(&self) -> bool {
        return self.double_spenders_total == self.double_spenders_revoked_total && self.double_spenders_total != 0; 
    }
    
    pub fn update(&mut self, stats: &Statistics) {
        self.potential_double_spender_max += stats.potential_double_spender_max;
        self.double_spenders_total += stats.double_spenders_total;
        self.double_spenders_revoked_total += stats.double_spenders_revoked_total;
        self.txns_total += stats.txns_total;
        self.txns_p2p_total += stats.txns_p2p_total;
        self.txns_p2m_total += stats.txns_p2m_total;
        self.txns_m2m_total += stats.txns_m2m_total;
        self.txns_b2p_total += stats.txns_b2p_total;
        self.txns_p2b_total += stats.txns_p2b_total;
        self.txns_b2m_total += stats.txns_b2m_total;
        self.txns_m2b_total += stats.txns_m2b_total;
        self.txns_b2b_total += stats.txns_b2b_total;
        self.txns_p2p_rejected_total += stats.txns_p2p_rejected_total;
        self.txns_p2m_rejected_total += stats.txns_p2m_rejected_total;
        self.txns_m2m_rejected_total += stats.txns_m2m_rejected_total;
        self.txns_b2m_rejected_total += stats.txns_b2m_rejected_total;
        self.txns_m2b_rejected_total += stats.txns_m2b_rejected_total;
        self.txns_b2p_rejected_total += stats.txns_b2p_rejected_total;
        self.txns_p2b_rejected_total += stats.txns_p2b_rejected_total;
        self.txns_b2b_rejected_total += stats.txns_b2b_rejected_total;
        self.txns_rejected_total += stats.txns_rejected_total;
        self.register_total += stats.register_total;
        self.synchronize_total += stats.synchronize_total;
        self.coins_total += stats.coins_total;
        self.coins_valid_total += stats.coins_valid_total;
        self.coins_double_spent_total += stats.coins_double_spent_total;
        self.double_spent_longest_life += stats.double_spent_longest_life;
        self.double_spent_most_txns += stats.double_spent_most_txns;
        self.coins_double_spent_recovered += stats.coins_double_spent_recovered;
        self.coins_double_spent_recovered_repeats += stats.coins_double_spent_recovered_repeats;
        self.validate_total += stats.validate_total;
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Default)]
pub struct ConsumerData {
    lifetime: usize, // in steps
    // Location native to Agent
    sync_probability: f64,
    sync_distribution: SupportedDistributions,
    p2m_probability: f64,
    p2m_distribution: SupportedDistributions,
    p2p_probability: f64,
    p2p_distribution: SupportedDistributions,
    double_spend_probability: f64,
    double_spend_distribution: SupportedDistributions,
    max_rejections: usize,
    move_distribution: SupportedDistributions,
    move_probability: f64,
    wids: usize,
    wid_low_watermark: usize,
    account_balance: usize,
    last_requested_step: usize,
    bank: usize,
}
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Default)]
pub struct MerchantData {
    lifetime: usize, // in steps
    // Location native to Agent
    // sync_probability: f64,
    // sync_distribution: SupportedDistributions,
    sync_frequency: usize, 
    account_balance: usize,
    last_tx_step: Option<usize>,
    bank: usize,
}
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Default)]
pub struct BankData {
    holding: Vec<Coin>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum AgentRole {
    Consumer(ConsumerData),
    Merchant(MerchantData),
    Bank(BankData),
}

impl AgentRole {
    pub fn is_consumer(&self) -> bool {
        match self {
            AgentRole::Consumer(_) => true,
            _ => false,
        }
    }
    pub fn is_merchant(&self) -> bool {
        match self {
            AgentRole::Merchant(_) => true,
            _ => false,
        }
    }
    pub fn is_bank(&self) -> bool {
        match self {
            AgentRole::Bank(_) => true,
            _ => false,
        }
    }
}
impl Default for AgentRole {
    fn default() -> Self {
        AgentRole::Consumer(ConsumerData::default())
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub struct AgentData {
    location: GraphVertexIndex,
    registered: bool,
    epoch: usize,
    coins: Vec<Coin>,
    pending: Vec<Event<EventData>>, // Events which should be fired off in the next gen phase
    role: AgentRole,
}
impl Default for AgentData {
    fn default() -> Self {
        AgentData {
            location: GraphVertexIndex(0),
            registered: false,
            epoch: 0,
            coins: vec![],
            pending: vec![],
            role: AgentRole::default(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Default)]
pub struct ResourceData {
    location: GraphVertexIndex,
    class: ResourceClass,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct PopulationAdd {
    data: AgentData,
    count: usize,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct PopulationDel {
    ids: Vec<usize>, // Agent ids
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct RegisterData {}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct SynchronizeData {}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct ValidateData {
    coins: Vec<Coin>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct ValidateResponseData {
    ok: Vec<usize>,          // coin.id
    counterfeit: Vec<usize>, // coin.id
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct GossipData {
    // updates source and, optionally, a target with state. Should not clobber newer state.
    epoch: usize,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct TrimData {
    coins: Vec<Coin>, // World updates; Agent self-trims
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct UpdateEpochData {
    revoked: usize, // Agent id.
}

// Must mirror AgentRoles
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum RequestRole {
    Consumer,
    Merchant,
    Bank,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct RequestTransactionData {
    amount: usize,
    epoch: usize, // Since we can't see it during apply.
    role: RequestRole,
    source_location: GraphVertexIndex 
}

// Any event with World as a destination is a world event.
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum EventData {
    // Agents
    Age(usize),
    Register(RegisterData),
    Synchronize(SynchronizeData), // Grant tickets in here.
    Validate(ValidateData),
    ValidateResponse(ValidateResponseData), // Returns bad coins
    Transact(TransactData),
    RejectedTransact,
    Gossip(GossipData),
    RequestTransaction(RequestTransactionData),
    Move(GraphVertexIndex),
    Trim(TrimData),
    // World/Operator
    UpdateEpoch(UpdateEpochData),
    UpdateStatistics(Statistics),
    // TODO: CoinSwap(CoinSwapData), // And max history len
    // TODO: To allow bank/online merchant/etc discovery, query for the agent, a descriptor, its epoch so the recipient can queue cross-view/cell transactions.
    // ----
    // TODO: QueryAgents(QueryAgentsData)
    // TODO: QueryAgentsResponse(QARData)
    // Population
    Arrive(PopulationAdd),
    Depart(PopulationDel),
    // Don't see any need for extra enum layers.
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum SupportedDistributions {
    Uniform,
    Pareto,
    Levy,
}
impl Default for SupportedDistributions {
    fn default() -> Self {
        SupportedDistributions::Uniform
    }
}


#[derive(Clone, Debug)]
pub struct Rngs {
    uniform: Box<rand::rngs::StdRng>,
    pareto: Box<rand::rngs::StdRng>,
    levy: Box<rand::rngs::StdRng>,
}
impl Rngs {
    pub fn new(rng_conf: &RngConfiguration) -> Rngs {
        Rngs {
            uniform: Box::new(SeedableRng::seed_from_u64(rng_conf.seed)),
            pareto: Box::new(SeedableRng::seed_from_u64(rng_conf.seed)),
            levy: Box::new(SeedableRng::seed_from_u64(rng_conf.seed)),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Distributions {
    levy: Box<levy_distr::Levy<f64>>,
    pareto: Box<rand_distr::Pareto<f64>>,
    uniform: Box<rand_distr::Uniform<f64>>,
}
impl Distributions {
    pub fn new(rng_conf: &RngConfiguration) -> Distributions {
        Distributions {
            levy: Box::new(Levy::<f64>::new(rng_conf.levy_min_step, rng_conf.levy_alpha).unwrap()),
            pareto: Box::new(Pareto::new(rng_conf.pareto_scale, rng_conf.pareto_shape).unwrap()),
            uniform: Box::new(Uniform::new::<f64, f64>(0., rng_conf.uniform_max)),
        }
    }
}

pub struct RngContext {
    rngs: Rngs,
    distributions: Distributions,
}
impl RngContext {
    pub fn new(rng_conf: &RngConfiguration) -> RngContext {
        RngContext {
            rngs: Rngs::new(&rng_conf),
            distributions: Distributions::new(&rng_conf),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Flights {
    levy: Flight<rand::rngs::StdRng, levy_distr::Levy<f64>, u64, 1>,
    pareto: Flight<rand::rngs::StdRng, rand_distr::Pareto<f64>, u64, 1>,
    uniform: Flight<rand::rngs::StdRng, rand_distr::Uniform<f64>, u64, 1>,
}

impl Flights {
    pub fn new(bounds: &[usize; 1], ctx: Box<RngContext>) -> Flights {
        Flights {
            uniform: Flight::<StdRng, Uniform<f64>, u64, 1>::new(
                ctx.rngs.uniform,
                ctx.distributions.uniform,
                [bounds[0].try_into().unwrap()],
            )
            .unwrap(),
            pareto: Flight::<StdRng, Pareto<f64>, u64, 1>::new(
                ctx.rngs.pareto,
                ctx.distributions.pareto,
                [bounds[0].try_into().unwrap()],
            )
            .unwrap(),
            levy: Flight::<StdRng, Levy<f64>, u64, 1>::new(
                ctx.rngs.levy,
                ctx.distributions.levy,
                [bounds[0].try_into().unwrap()],
            )
            .unwrap(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SimulatorHelpers {
    rng: Box<rand::rngs::StdRng>,
    distributions: Distributions,
    flights: Flights,
}
impl SimulatorHelpers {
    pub fn new(bounds: usize, rng_config: &RngConfiguration) -> SimulatorHelpers {
        // Setup all the rngs
        let flight_ctx = Box::new(RngContext::new(&rng_config));
        // TODO: Separate configs!
        // Create FlightRng and RngConfig
        let mut rcfg = rng_config.clone();
        rcfg.uniform_max = 1.0;
        let rng_context = Box::new(RngContext::new(&rcfg));
        SimulatorHelpers {
            rng: rng_context.rngs.uniform,
            distributions: Distributions {
                levy: rng_context.distributions.levy,
                pareto: rng_context.distributions.pareto,
                uniform: rng_context.distributions.uniform,
            },
            flights: Flights::new(&[bounds], flight_ctx),
        }
    }
}
pub trait SimulationTools {
    fn probability_check(&mut self, distribution: &SupportedDistributions, threshold: f64) -> bool;
    fn get_uniform(&mut self, base: usize, max: usize) -> usize;
    fn relocate(&mut self, graph: &SimulationGraph, distribution: &SupportedDistributions, source: &GraphVertexIndex) -> GraphVertexIndex;
}

impl SimulationTools for SimulatorHelpers {
    fn relocate(
        &mut self,
        graph: &SimulationGraph, 
        distribution: &SupportedDistributions, 
        source: &GraphVertexIndex
    ) -> GraphVertexIndex {
        let mut dst: [u64; 1] = [0];
        match distribution {
            SupportedDistributions::Levy => {
                self.flights.levy.step_graph(graph.neighbours(source), &mut dst);
            }
            SupportedDistributions::Uniform => {
                self.flights.uniform.step_graph(graph.neighbours(source), &mut dst);
            }
            SupportedDistributions::Pareto => {
                self.flights.pareto.step_graph(graph.neighbours(source),&mut dst);
            }
        }
        GraphVertexIndex(dst[0].try_into().unwrap())
    }
    fn get_uniform(&mut self, base: usize, max: usize) -> usize {
        if max <= base {
            eprintln!("get_uniform: base >= max");
            return 0;
        }
        let range = Uniform::from(base..max);
        range.sample(&mut self.rng)
    }

    fn probability_check(&mut self, distribution: &SupportedDistributions, threshold: f64) -> bool {
        match distribution {
            SupportedDistributions::Uniform => {
                self.distributions.uniform.sample(&mut self.rng) <= threshold
            }
            SupportedDistributions::Pareto => {
                self.distributions.pareto.sample(&mut self.rng) <= threshold
            }
            SupportedDistributions::Levy => {
                self.distributions.levy.sample(&mut self.rng) <= threshold
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct RngConfiguration {
    seed: u64,
    levy_min_step: f64,
    levy_alpha: f64,
    pareto_scale: f64,
    pareto_shape: f64,
    uniform_max: f64,
}

impl RngConfiguration {
    pub fn new(
        seed: u64,
        levy_min_step: f64,
        levy_alpha: f64,
        pareto_scale: f64,
        pareto_shape: f64,
        uniform_max: f64,
    ) -> Self {
        RngConfiguration {
            seed,
            levy_min_step,
            levy_alpha,
            pareto_scale,
            pareto_shape,
            uniform_max,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct WorldData {
    // World
    step: usize, // GlobalTick event
    graph_size: usize,
    graph: SimulationGraph,
    resources: Vec<ResourceData>,
    pub statistics: Statistics,
    // Operator data
    epoch: usize, // current epoch
    epochs: Vec<SyncState>,
    last_coin_index: usize,              // for minting
    coin_map: HashMap<usize, CoinState>, // for keeping historical data view
    banks: Vec<usize>,                   // List of bank ids
    pending: Vec<Event<EventData>>,      // Events which should be fired off in the next gen phase
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ViewData {
    id: usize,
    step: usize, // GlobalTick event
    bounds: usize,
    resource: ResourceData, // Just one for now.
    // We could copy WorldData here, but it seems less messy to just copy what we need.
    epoch: usize,
    epochs: Vec<SyncState>,
    graph: SimulationGraph
}


#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Default)]
pub struct NetworkData {
    online_probability: f64,
    online_distribution: SupportedDistributions,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub struct AgentConfiguration {
    class: AgentRole,
    count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ResourceClass {
    Network(NetworkData),
}
impl Default for ResourceClass {
    fn default() -> Self {
        ResourceClass::Network(NetworkData::default())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub struct ResourceConfiguration {
    class: ResourceClass,
    count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Configuration {
    rng_configuration: RngConfiguration,
    //display_configuration: DisplayConfiguration,
    bounds: (usize, usize),
    num_steps: usize,

    agents: Vec<AgentConfiguration>,
    resources: Vec<ResourceConfiguration>,
    // TODO(wad) Add online grid region mapping.
    // TODO(wad) Add merchant grid region mapping.
}

impl Configuration {
    pub fn from_file(path: &String) -> Configuration {
        let file_content = fs::read_to_string(path).expect("GridSim: error reading configuration");
        serde_json::from_str::<Configuration>(&file_content)
            .expect("GridSim: failed to parse configuration file")
        /*
        print!(
            "{}",
            serde_json::to_string_pretty(&payload).expect("GridSim: error printing cfg")
        );
        */
    }
}

fn main() {
    let graph: SimulationGraph = SimulationGraph::new(
        "/home/vladyslav/VSCodeProjects/briolette/src/simulation/briolettesim/graphs/barabasi_albert_test.txt"
    ).unwrap();
    // Use configured seed as root to spawn off all rngs.
    let mut rng: StdRng = SeedableRng::seed_from_u64(100);
    let mgr_seed = rng.gen::<u64>();
    // NYC 29,729/sqmil;  300.6 sq mi= 17.33*17.33 (0xuki: square grid most likely)
    // Let's aim smaller.  10*10 = 100, so 2972900 
    // Let's aim smaller.  3 * 3 = 9 so 297290
    let total_vertices_in_graph = graph.adjacency_list.len();
    println!("{total_vertices_in_graph}");

    let helper = SimulatorHelpers::new(
        total_vertices_in_graph.clone(),
        &RngConfiguration::new(
            mgr_seed, 
            3.0, 
            1.8, 
            2.0,
            2.0, 
            1.0
        ),
    ); 

    // (0xuki: double check the numbers here)
    let num_consumers = 80; //237832; // 2972900; // Full NYC pop 18867000; // 139; // 10000;  // Scaled NYC pop
    let num_double_spenders = 20;
    let num_merchants = 50;
    let num_banks = 5;

    // Setup init stats
    let mut upstats = Statistics::default();

    let default_balance = 500;
    // Create enough coinage for every consumer to have $500 in the bank to get started.
    let mut coinage = Vec::new();
    // Use a map so we can remove these easily before we simulate.
    let mut bank_coins: HashMap<usize, Vec<Coin>> = HashMap::new();
    let mut coin_index = 0;
    for c in 0..(num_consumers * default_balance) {
        let b = c % num_banks; // (0xuki: bank id)
        coinage.push(Coin {
            id: coin_index,
            value: 1, // all coins in the simulation have value 1!
            copied: false,
            history: vec![usize::MAX, b], // mint is usize::MAX
            tx_history: vec![0, 1],       // Don't need randomness since this is controlled.
            step_history: vec![0, 0],
        });
        if let Some(treasury) = bank_coins.get_mut(&b) {
            treasury.push(coinage.iter().last().unwrap().clone());
        } else {
            bank_coins.insert(b, vec![coinage.iter().last().unwrap().clone()]);
        }
        coin_index += 1;
    }
    upstats.coins_total = coinage.len();
    upstats.coins_valid_total = coinage.len();

    let mut mgr = Manager::new(
        Simulator::new(mgr_seed, helper),
        SimulationPopulation::new(),
        WorldData {
            step: 0,
            graph_size: total_vertices_in_graph,
            graph,
            resources: vec![],
            statistics: Statistics::default(),
            // Operator init
            epoch: 0,
            epochs: vec![SyncState::default()],
            last_coin_index: coin_index,
            coin_map: coinage
                .into_iter()
                .enumerate()
                .map(|(i, v)| (i, CoinState::new(v)))
                .collect(),
            banks: (0..num_banks).collect(), // Add banks first. TODO: track then via Register() later.
            pending: vec![],
        },
    );
    //print!("Manager: {}", serde_json::to_string_pretty(&mgr).unwrap());
    for _ in 0..4 {
        let client_seed = rng.gen::<u64>();
        mgr.add_client(Box::new(LocalSimulationClient::new(Simulator::new(
            client_seed,
            SimulatorHelpers::new(
                total_vertices_in_graph.clone(),
                &RngConfiguration::new(client_seed, 3.0, 1.8, 2.0, 2.0, 1.0),
            ),
        ))));
    }
    // Create a few banks and expect their IDs to be 0..num_banks
    for b in 0..num_banks {
        let idx = rng.gen_range(0..mgr.world().graph_size);
        mgr.enqueue(
            Address::NoAddress,
            Address::Population,
            vec![EventData::Arrive(PopulationAdd {
                data: AgentData {
                    location: GraphVertexIndex(idx),
                    registered: true,
                    epoch: 0,
                    coins: bank_coins.remove(&b).unwrap(),
                    pending: vec![],
                    role: AgentRole::Bank(BankData { holding: vec![] }),
                },
                count: 1,
            })],
        );
    }

    // Randomly place 10k consumers
    for c in 0..num_consumers {
        let idx = rng.gen_range(0..mgr.world().graph_size);
        mgr.enqueue(
            Address::NoAddress,
            Address::Population,
            vec![EventData::Arrive(PopulationAdd {
                data: AgentData {
                    location: GraphVertexIndex(idx),
                    registered: false,
                    epoch: 0,
                    coins: vec![],
                    pending: vec![],
                    role: AgentRole::Consumer(ConsumerData {
                        lifetime: 43800, // 5 years in hours for phone lifetime.
                        sync_probability: 0.01,
                        sync_distribution: SupportedDistributions::Uniform,
                        p2m_probability: 0.2,
                        p2m_distribution: SupportedDistributions::Uniform,
                        p2p_probability: 0.6,
                        p2p_distribution: SupportedDistributions::Uniform,
                        double_spend_probability: 0.0,
                        double_spend_distribution: SupportedDistributions::Uniform,
                        max_rejections: 5,
                        move_distribution: SupportedDistributions::Uniform,
                        move_probability: 0.9,
                        /*(
                        step_period: 24,
                        max_transactions_per_period: 2,
                        */
                        wids: 8,
                        wid_low_watermark: 2,
                        account_balance: default_balance,
                        last_requested_step: 0,
                        // (0xuki) TODO: change bank assignment procedure to account for distance to the bank!
                        bank: c % num_banks, // ensure we match the balance with the coins. TODO: Register() to get bank and balance.
                    }),
                },
                count: 1,
            })],
        );
    }

    // Randomly place double spenders
    for _ in 0..num_double_spenders {
        let idx = rng.gen_range(0..mgr.world().graph_size);
        mgr.enqueue(
            Address::NoAddress,
            Address::Population,
            vec![EventData::Arrive(PopulationAdd {
                data: AgentData {
                    location: GraphVertexIndex(idx),
                    registered: false,
                    epoch: 0,
                    coins: vec![],
                    pending: vec![],
                    role: AgentRole::Consumer(ConsumerData {
                        lifetime: 43800, // 5 years in hours for phone lifetime.
                        sync_probability: 0.0,
                        sync_distribution: SupportedDistributions::Uniform,
                        p2m_probability: 1.0,
                        p2m_distribution: SupportedDistributions::Uniform,
                        p2p_probability: 1.0,
                        p2p_distribution: SupportedDistributions::Uniform,
                        double_spend_probability: 1.0,
                        double_spend_distribution: SupportedDistributions::Uniform,
                        max_rejections: 5,
                        move_distribution: SupportedDistributions::Uniform,
                        move_probability: 1.0,
                        /*
                        step_period: 24,
                        max_transactions_per_period: 72,
                        */
                        wids: 8,
                        wid_low_watermark: 2,
                        account_balance: default_balance,
                        last_requested_step: 0,
                        bank: rng.gen_range(0..num_banks),
                    }),
                },
                count: 1,
            })],
        );
    }

    // Randomly place double spenders in the future
    for _ in 0..num_double_spenders {
        let idx = rng.gen_range(0..mgr.world().graph_size);
        mgr.enqueue_delayed(
            Address::NoAddress,
            Address::Population,
            vec![EventData::Arrive(PopulationAdd {
                data: AgentData {
                    location: GraphVertexIndex(idx),
                    registered: false,
                    epoch: 0,
                    coins: vec![],
                    pending: vec![],
                    role: AgentRole::Consumer(ConsumerData {
                        lifetime: 43800, // 5 years in hours for phone lifetime.
                        sync_probability: 0.0,
                        sync_distribution: SupportedDistributions::Uniform,
                        p2m_probability: 1.0,
                        p2m_distribution: SupportedDistributions::Uniform,
                        p2p_probability: 1.0,
                        p2p_distribution: SupportedDistributions::Uniform,
                        double_spend_probability: 1.0,
                        double_spend_distribution: SupportedDistributions::Uniform,
                        max_rejections: 5,
                        move_distribution: SupportedDistributions::Uniform,
                        move_probability: 1.0,
                        /*
                        step_period: 24,
                        max_transactions_per_period: 72,
                        */
                        wids: 8,
                        wid_low_watermark: 2,
                        account_balance: default_balance,
                        last_requested_step: 0,
                        bank: rng.gen_range(0..num_banks),
                    }),
                },
                count: 1,
            })],
            15,
        );
    }

    upstats.double_spenders_total = num_double_spenders * 2;
    // Update the states
    mgr.enqueue(
        Address::NoAddress,
        Address::World,
        vec![EventData::UpdateStatistics(upstats)],
    );

    // Randomly place consumers/115 merchants
    for _ in 0..num_merchants {
        let idx = rng.gen_range(0..(mgr.world().graph_size));
        mgr.enqueue(
            Address::NoAddress,
            Address::Population,
            vec![EventData::Arrive(PopulationAdd {
                data: AgentData {
                    location: GraphVertexIndex(idx),
                    registered: false,
                    epoch: 0,
                    coins: vec![],
                    pending: vec![],
                    role: AgentRole::Merchant(MerchantData {
                        lifetime: 183960, // 21 years in hours -- avg lifespan of company S&P.
                        sync_frequency: 8, // every `sync_frequency` hours make a transfer to the bank -> sync as a result!
                        // sync_probability: 0.01, 
                        // sync_distribution: SupportedDistributions::Uniform,
                        account_balance: 0,
                        last_tx_step: None,
                        bank: rng.gen_range(0..num_banks),
                    }),
                },
                count: 1,
            })],
        );
    }

    // Install network probability in every grid point.
    for idx in 0..total_vertices_in_graph {
        mgr.world().resources.push(ResourceData {
            location: GraphVertexIndex(idx),
            class: ResourceClass::Network(NetworkData {
                online_probability: 0.8,
                online_distribution: SupportedDistributions::Uniform,
            }),
        });
    }

    mgr.register_observer(1, &observe);
    mgr.register_observer(1, &check_exit_conditions_and_print_results_to_file);
    mgr.run(4000, "/home/vladyslav/VSCodeProjects/briolette/src/simulation/briolettesim/results/experiment_results.txt");
}

fn observe(step: usize, world: &WorldData, _pop: &SimulationPopulation<Simulator>, _end_simulation_flag: &mut bool, _file_path: &str) {
    println!(
        "[{}] --------------------------------------------------------------------",
        step
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&world.statistics).unwrap()
    );
    /*
    let num_con = pop
        .agents
        .iter()
        .filter(|agt| match agt.1.data.role {
            AgentRole::Consumer(_) => true,
            _ => false,
        })
        .count();
    let num_mer = pop
        .agents
        .iter()
        .filter(|agt| match agt.1.data.role {
            AgentRole::Merchant(_) => true,
            _ => false,
        })
        .count();
    println!("[{}] consumers: {} merchants: {}", step, num_con, num_mer);
    */
}


fn check_exit_conditions_and_print_results_to_file(
    step: usize, 
    world: &WorldData, 
    _pop: &SimulationPopulation<Simulator>,
    end_simulation_flag: &mut bool, 
    file_path: &str
) {
    if (world.statistics.double_spenders_total == world.statistics.double_spenders_revoked_total) {
        *end_simulation_flag = true;
        // time to exit; write important information about the experiment to the file, and flag that we need to move on (???)
        // need to pass a flag inside for "exiting" the simulation for current parameters I suppose...
        let mut file = match File::create(file_path) {
            Ok(handle) => handle,
            Err(e) => panic!("{}", e),
        };
        
        // How many simulation steps have passed since initial double-spending fork?
        for &item in &world.statistics.double_spent_life_measurements {
            match write!(file, "{} ", item) {
                Ok(_) => (),
                Err(e) => panic!("{}", e) 
            }
        }
        match writeln!(file) {
            Ok(_) => (),
            Err(e) => panic!("{}", e) 
        };

        // How many transactions have passed since initial double-spending fork?
        for &item in &world.statistics.double_spent_txs_measurements {
            match write!(file, "{} ", item) {
                Ok(_) => (),
                Err(e) => panic!("{}", e) 
            }
        }
        match writeln!(file) {
            Ok(_) => (),
            Err(e) => panic!("{}", e) 
        };

        // Step when all cheaters have been caught
        match writeln!(file, "{}", step) {
            Ok(_) => (),
            Err(e) => panic!("{}", e) 
        };

        



        println!("Wrote stats to the file!");
        
    }
}