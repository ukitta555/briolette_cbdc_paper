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
pub mod utils;

use std::{fs::{File, OpenOptions}, io::{self, BufRead}, time::{SystemTime, UNIX_EPOCH, Instant}};
use absim::graph_utils::{GraphVertexIndex, SimulationGraph};
use rand::{prelude::*};
use rand::{rngs::StdRng, SeedableRng};
use serde::{Deserialize, Serialize};
use simulator::Simulator;
use utils::split_population;
use std::collections::HashMap;
use std::fs::{self};
use std::io::Write;
use clap::{Parser, Subcommand};
use std::path::Path;
use chrono::Local;
use rayon::prelude::*;
use std::process;

use absim::clients::LocalSimulationClient;
use absim::extras::SimulationPopulation;
use absim::{
    Address, Event, Manager, ManagerInterface,
};
use levy_distr::Levy;
use rand_distr::{Pareto, Uniform};
use rand_flight::Flight;


#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

// cargo run -- predefined
// cargo run -- sobol --params-file /path/to/sobol_params.txt --repeat 100 --population-size 1000
#[derive(Subcommand)]
enum Commands {
    /// Run experiments with predefined configurations
    Predefined,
    /// Run experiments with Sobol parameter sampling
    /// 
    /// This command runs experiments using Sobol sequence sampling for parameter space exploration.
    /// It reads parameter combinations from a file and runs multiple experiments with different
    /// parameter values to analyze system behavior across a wide range of configurations.
    /// 
    /// The parameters file should contain space-separated values for:
    /// - move_probability: Probability of agents moving between locations
    /// - p2p_probability: Probability of peer-to-peer transactions
    /// - p2m_probability: Probability of peer-to-merchant transactions
    /// - ratio_double_spenders_to_honest: Ratio of malicious to honest agents
    Sobol {
        /// Path to the Sobol parameters file containing space-separated parameter values
        #[arg(short, long)]
        params_file: String,
        
        /// Number of times to repeat each experiment for statistical significance
        #[arg(short, long, default_value_t = 100)]
        repeat: usize,
        
        /// Total number of user agents in the simulation
        #[arg(short = 's', long, default_value_t = 1000)]
        population_size: usize,
        
        /// Maximum number of parallel threads to use (default: auto-detect based on memory)
        #[arg(short = 't', long)]
        max_threads: Option<usize>,
    },
}

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
#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone, Default)]
pub struct SyncState {
    id: usize,
    revocation: Vec<usize>, // list of agent ids for now.
    quarantine: Vec<usize>, // group ids to manage WID issuance?
    step: usize,            // Create at what step
}

#[derive(Debug, PartialEq, Clone)]
pub struct TransactionCoin {
    coin: Coin,
    copy: bool,   // true to double spend
    popped: bool, // True if the coin has already been removed from the sender.
}
#[derive(Debug, PartialEq, Clone)]
pub struct TransactData {
    coins: Vec<TransactionCoin>, // double spending is literal this way.
}

#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
pub struct EpochSampleStats {
    mean: f64,
    standard_deviation: f64,
    max_diff: usize
}

#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
pub struct Statistics {
    potential_double_spender_max: usize,
    double_spenders_total: usize,
    double_spenders_revoked_total: usize,
    total_people: usize,
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
    double_spent_life_measurements: Vec<(usize, usize)>, // (epoch, life)
    #[serde(skip_serializing)]
    double_spent_txs_measurements: Vec<(usize, usize)>, // (epoch, txs)
    #[serde(skip_serializing)]  
    global_to_local_epoch_diffs: Vec<EpochSampleStats>,
    #[serde(skip_serializing)]
    std_local_epoch_diffs: Vec<f64>,
    #[serde(skip_serializing)]
    ratios_of_double_spent_coins: Vec<f64>,
    #[serde(skip_serializing)]
    ratio_of_double_spenders_caught: Vec<f64>,
    #[serde(skip_serializing)]
    caught_ratios_epsilon_stop: Vec<f64>   
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
        self.total_people += stats.total_people;
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

#[derive(PartialEq, Clone)]
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

#[derive(Debug, PartialEq, Clone, Default)]
pub struct ResourceData {
    location: GraphVertexIndex,
    class: ResourceClass,
}

#[derive(PartialEq, Clone)]
pub struct PopulationAdd {
    data: AgentData,
}

#[derive(Debug, PartialEq, Clone)]
pub struct PopulationDel {
    ids: Vec<usize>, // Agent ids
}

#[derive(Debug, PartialEq, Clone)]
pub struct RegisterData {}

#[derive(Debug, PartialEq, Clone)]
pub struct SynchronizeData {}

#[derive(Debug, PartialEq, Clone)]
pub struct ValidateData {
    coins: Vec<Coin>,
}

#[derive(Debug, PartialEq, Clone)]
pub struct ValidateResponseData {
    ok: Vec<usize>,          // coin.id
    counterfeit: Vec<usize>, // coin.id
}

#[derive(Debug, PartialEq, Clone)]
pub struct GossipData {
    // updates source and, optionally, a target with state. Should not clobber newer state.
    epoch: usize,
}

#[derive(Debug, PartialEq, Clone)]
pub struct TrimData {
    coins: Vec<Coin>, // World updates; Agent self-trims
}

#[derive(Debug, PartialEq, Clone)]
pub struct UpdateEpochData {
    revoked: usize, // Agent id.
}

// Must mirror AgentRoles
#[derive(Debug, PartialEq, Clone)]
pub enum RequestRole {
    Consumer,
    Merchant,
    Bank,
}

#[derive(Debug, PartialEq, Clone)]
pub struct RequestTransactionData {
    amount: usize,
    epoch: usize, // Since we can't see it during apply.
    role: RequestRole,
    source_location: GraphVertexIndex 
}

// Any event with World as a destination is a world event.
#[derive(PartialEq, Clone)]
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

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
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

#[derive(Clone)]
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
    deposit_limit: usize,
    ticket_refill: usize,
}

#[derive(Debug, Clone)]
pub struct ViewData {
    id: usize,
    step: usize, // GlobalTick event
    bounds: usize,
    deposit_limit: usize,
    tickets_to_give_for_refill: usize,
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

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct AgentConfiguration {
    class: AgentRole,
    count: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum ResourceClass {
    Network(NetworkData),
}
impl Default for ResourceClass {
    fn default() -> Self {
        ResourceClass::Network(NetworkData::default())
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
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

#[derive(Clone, Debug)]
pub enum ExperimentCategory {
    ThreatLevel,
    SyncParams,
    Sobol,
}

#[derive(Clone, Debug)]
pub struct ExperimentConfig {
    ratio_double_spenders_to_honest: f64,
    top_up_amount: usize,
    merchants: usize,
    banks: usize,
    graph_file: String,
    p2m_probability: f64,
    p2p_probability: f64,
    move_probability: f64,
    random_sync_probability: f64,
    merchant_sync_frequency: usize,
    tickets_given_right_away: usize,
    tickets_lower_bound_to_sync: usize,
    account_balance: usize,
    model: Model,
    graph_config: GraphConfig,
    category: ExperimentCategory,
}

#[derive(Clone, Debug)]
enum Model {
    Urban,
    Rural,
}

#[derive(Clone, Debug)]
struct BarabasiConfig {
    size: usize,
    param: usize
}


#[derive(Clone, Debug)]
struct WattsConfig {
    size: usize,
    param: usize
}

#[derive(Clone, Debug)]
enum GraphConfig {
    Watts(WattsConfig),
    Barabasi(BarabasiConfig),
}

fn create_experiment_directory(experiment_type: &str, population_size: u64) -> io::Result<String> {
    let base_dir = match experiment_type {
        "predefined" => "/home/fanlgrp/Projects/briolette_cbdc_paper/src/simulation/briolettesim/results/predefined_experiments",
        "sobol" => "/home/fanlgrp/Projects/briolette_cbdc_paper/src/simulation/briolettesim/results/sobol_experiments",
        _ => panic!("Invalid experiment type"),
    };

    // Create base directory if it doesn't exist
    fs::create_dir_all(base_dir)?;

    // Create a timestamped directory with population size
    let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
    let dir_name = format!("{}/experiment_{}_{}k_actors", base_dir, timestamp, population_size / 1000);
    fs::create_dir_all(&dir_name)?;

    // Create subdirectories for different model types
    let urban_dir = format!("{}/urban", dir_name);
    let rural_dir = format!("{}/rural", dir_name);
    fs::create_dir_all(&urban_dir)?;
    fs::create_dir_all(&rural_dir)?;

    Ok(dir_name)
}

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() -> io::Result<()> {
    let program_start = Instant::now();
    println!("Program started at: {:?}", program_start);
    
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();
    let cli = Cli::parse();

    match cli.command {
        Commands::Predefined => {
            let population_size = 30000;
            let repeat_experiment_number_of_times = 1;
            let mut experiments: HashMap<usize, Vec<ExperimentConfig>> = HashMap::new();
            let experiment_dir = create_experiment_directory("predefined", population_size as u64)?;

            // Add your predefined experiments here
            let predefined_experiments = vec![
                ExperimentConfig { // 1:1
                    ratio_double_spenders_to_honest: 1.0,
                    top_up_amount: 10, 
                    merchants: 30, 
                    banks: 10,
                    graph_file: String::from("/home/fanlgrp/Projects/briolette_cbdc_paper/src/simulation/briolettesim/graphs/barabasi_albert_8_m.txt"),
                    p2m_probability: 0.2, 
                    p2p_probability: 0.6, 
                    move_probability: 0.2,  
                    random_sync_probability: 0.01,  
                    merchant_sync_frequency: 8,
                    tickets_given_right_away: 8,
                    tickets_lower_bound_to_sync: 2,
                    account_balance: 500,
                    model: Model::Urban,
                    graph_config: GraphConfig::Barabasi(BarabasiConfig {
                        size: 64, 
                        param: 12,
                    }),
                    category: ExperimentCategory::ThreatLevel,
                },
                ExperimentConfig { // 2:1
                    ratio_double_spenders_to_honest: 0.5,
                    top_up_amount: 10,
                    merchants: 30, 
                    banks: 10,
                    graph_file: String::from("/home/fanlgrp/Projects/briolette_cbdc_paper/src/simulation/briolettesim/graphs/barabasi_albert_8_m.txt"),
                    p2m_probability: 0.2, 
                    p2p_probability: 0.6, 
                    move_probability: 0.2,  
                    random_sync_probability: 0.01,  
                    merchant_sync_frequency: 8,
                    tickets_given_right_away: 8,
                    tickets_lower_bound_to_sync: 2,
                    account_balance: 500,
                    model: Model::Urban,
                    graph_config: GraphConfig::Barabasi(BarabasiConfig {
                        size: 64, 
                        param: 12,
                    }),
                    category: ExperimentCategory::ThreatLevel,
                },
                ExperimentConfig { // 4:1
                    ratio_double_spenders_to_honest: 0.25,
                    top_up_amount: 10, 
                    merchants: 30, 
                    banks: 10,
                    graph_file: String::from("/home/fanlgrp/Projects/briolette_cbdc_paper/src/simulation/briolettesim/graphs/barabasi_albert_8_m.txt"),
                    p2m_probability: 0.2, 
                    p2p_probability: 0.6, 
                    move_probability: 0.2,  
                    random_sync_probability: 0.01,  
                    merchant_sync_frequency: 8,
                    tickets_given_right_away: 8,
                    tickets_lower_bound_to_sync: 2,
                    account_balance: 500,
                    model: Model::Urban,
                    graph_config: GraphConfig::Barabasi(BarabasiConfig {
                        size: 64, 
                        param: 12,
                    }),
                    category: ExperimentCategory::ThreatLevel,
                },
                ExperimentConfig { // 10:1
                    ratio_double_spenders_to_honest: 0.1,
                    top_up_amount: 10, 
                    merchants: 30, 
                    banks: 10,
                    graph_file: String::from("/home/fanlgrp/Projects/briolette_cbdc_paper/src/simulation/briolettesim/graphs/barabasi_albert_8_m.txt"),
                    p2m_probability: 0.2, 
                    p2p_probability: 0.6, 
                    move_probability: 0.2,  
                    random_sync_probability: 0.01,  
                    merchant_sync_frequency: 8,
                    tickets_given_right_away: 8,
                    tickets_lower_bound_to_sync: 2,
                    account_balance: 500,
                    model: Model::Urban,
                    graph_config: GraphConfig::Barabasi(BarabasiConfig {
                        size: 64, 
                        param: 12,
                    }),
                    category: ExperimentCategory::ThreatLevel,
                },
                ExperimentConfig { // 2:1, big top-up, merchants sync every 24 hours, 50 tickets > slowest sync
                    ratio_double_spenders_to_honest: 0.5,
                    top_up_amount: 50, 
                    merchants: 30, 
                    banks: 10,
                    graph_file: String::from("/home/fanlgrp/Projects/briolette_cbdc_paper/src/simulation/briolettesim/graphs/barabasi_albert_8_m.txt"),
                    p2m_probability: 0.2, 
                    p2p_probability: 0.5, 
                    move_probability: 0.3,  
                    random_sync_probability: 0.01,  
                    merchant_sync_frequency: 24,
                    tickets_given_right_away: 50,
                    tickets_lower_bound_to_sync: 2,
                    account_balance: 500,
                    model: Model::Urban,
                    graph_config: GraphConfig::Barabasi(BarabasiConfig {
                        size: 64, 
                        param: 12,
                    }),
                    category: ExperimentCategory::SyncParams,
                },
                ExperimentConfig { // 2:1, medium top-up, merchants sync every 12 hours, 20 tickets  > slower sync
                    ratio_double_spenders_to_honest: 0.5,
                    top_up_amount: 20, 
                    merchants: 30, 
                    banks: 10,
                    graph_file: String::from("/home/fanlgrp/Projects/briolette_cbdc_paper/src/simulation/briolettesim/graphs/barabasi_albert_8_m.txt"),
                    p2m_probability: 0.2, 
                    p2p_probability: 0.5, 
                    move_probability: 0.3,  
                    random_sync_probability: 0.01,  
                    merchant_sync_frequency: 12,
                    tickets_given_right_away: 20,
                    tickets_lower_bound_to_sync: 2,
                    account_balance: 500,
                    model: Model::Urban,
                    graph_config: GraphConfig::Barabasi(BarabasiConfig {
                        size: 64, 
                        param: 12,
                    }),
                    category: ExperimentCategory::SyncParams,
                },
                ExperimentConfig { // 2:1, small top-up, merchants sync every 8 hours, 8 tickets > quick sync 
                    ratio_double_spenders_to_honest: 0.5,
                    top_up_amount: 5, 
                    merchants: 30, 
                    banks: 10,
                    graph_file: String::from("/home/fanlgrp/Projects/briolette_cbdc_paper/src/simulation/briolettesim/graphs/barabasi_albert_8_m.txt"),
                    p2m_probability: 0.2, 
                    p2p_probability: 0.5, 
                    move_probability: 0.3,  
                    random_sync_probability: 0.01,  
                    merchant_sync_frequency: 8,
                    tickets_given_right_away: 8,
                    tickets_lower_bound_to_sync: 2,
                    account_balance: 500,
                    model: Model::Urban,
                    graph_config: GraphConfig::Barabasi(BarabasiConfig {
                        size: 64, 
                        param: 12,
                    }),
                    category: ExperimentCategory::SyncParams,
                },
                // RURAL SCENARIO STARTS HERE
                ExperimentConfig { // 1:1
                    ratio_double_spenders_to_honest: 1.0,
                    top_up_amount: 10, 
                    merchants: 30, 
                    banks: 5,
                    graph_file: String::from("/home/fanlgrp/Projects/briolette_cbdc_paper/src/simulation/briolettesim/graphs/watts-strogatz-5-connections.txt"),
                    p2m_probability: 0.2, 
                    p2p_probability: 0.6, 
                    move_probability: 0.2,  
                    random_sync_probability: 0.01,  
                    merchant_sync_frequency: 8,
                    tickets_given_right_away: 8,
                    tickets_lower_bound_to_sync: 2,
                    account_balance: 500,
                    model: Model::Rural,
                    graph_config: GraphConfig::Watts(WattsConfig {
                        size: 64, 
                        param: 15,
                    }),
                    category: ExperimentCategory::ThreatLevel,
                },
                ExperimentConfig { // 2:1
                    ratio_double_spenders_to_honest: 0.5,
                    top_up_amount: 10,
                    merchants: 30, 
                    banks: 5,
                    graph_file: String::from("/home/fanlgrp/Projects/briolette_cbdc_paper/src/simulation/briolettesim/graphs/watts-strogatz-5-connections.txt"),
                    p2m_probability: 0.2, 
                    p2p_probability: 0.6, 
                    move_probability: 0.2,  
                    random_sync_probability: 0.01,  
                    merchant_sync_frequency: 8,
                    tickets_given_right_away: 8,
                    tickets_lower_bound_to_sync: 2,
                    account_balance: 500,
                    model: Model::Rural,
                    graph_config: GraphConfig::Watts(WattsConfig {
                        size: 64, 
                        param: 15,
                    }),
                    category: ExperimentCategory::ThreatLevel,
                },
                ExperimentConfig { // 4:1
                    ratio_double_spenders_to_honest: 0.25,
                    top_up_amount: 10, 
                    merchants: 30, 
                    banks: 5,
                    graph_file: String::from("/home/fanlgrp/Projects/briolette_cbdc_paper/src/simulation/briolettesim/graphs/watts-strogatz-5-connections.txt"),
                    p2m_probability: 0.2, 
                    p2p_probability: 0.6, 
                    move_probability: 0.2,  
                    random_sync_probability: 0.01,  
                    merchant_sync_frequency: 8,
                    tickets_given_right_away: 8,
                    tickets_lower_bound_to_sync: 2,
                    account_balance: 500,
                    model: Model::Rural,
                    graph_config: GraphConfig::Watts(WattsConfig {
                        size: 64, 
                        param: 15,
                    }),
                    category: ExperimentCategory::ThreatLevel,
                },
                ExperimentConfig { // 10:1
                    ratio_double_spenders_to_honest: 0.1,
                    top_up_amount: 10, 
                    merchants: 30, 
                    banks: 5,
                    graph_file: String::from("/home/fanlgrp/Projects/briolette_cbdc_paper/src/simulation/briolettesim/graphs/watts-strogatz-5-connections.txt"),
                    p2m_probability: 0.2, 
                    p2p_probability: 0.6, 
                    move_probability: 0.2,  
                    random_sync_probability: 0.01,  
                    merchant_sync_frequency: 8,
                    tickets_given_right_away: 8,
                    tickets_lower_bound_to_sync: 2,
                    account_balance: 500,
                    model: Model::Rural,
                    graph_config: GraphConfig::Watts(WattsConfig {
                        size: 64, 
                        param: 15,
                    }),
                    category: ExperimentCategory::ThreatLevel,
                },
                ExperimentConfig { // 2:1, big top-up, merchants sync every 24 hours, 50 tickets > slowest sync
                    ratio_double_spenders_to_honest: 0.5,
                    top_up_amount: 50, 
                    merchants: 30, 
                    banks: 5,
                    graph_file: String::from("/home/fanlgrp/Projects/briolette_cbdc_paper/src/simulation/briolettesim/graphs/watts-strogatz-5-connections.txt"),
                    p2m_probability: 0.2, 
                    p2p_probability: 0.6, 
                    move_probability: 0.2,  
                    random_sync_probability: 0.01,  
                    merchant_sync_frequency: 24,
                    tickets_given_right_away: 50,
                    tickets_lower_bound_to_sync: 2,
                    account_balance: 500,
                    model: Model::Rural,
                    graph_config: GraphConfig::Watts(WattsConfig {
                        size: 64, 
                        param: 15,
                    }),
                    category: ExperimentCategory::SyncParams,
                },
                ExperimentConfig { // 2:1, medium top-up, merchants sync every 12 hours, 20 tickets  > slower sync
                    ratio_double_spenders_to_honest: 0.5,
                    top_up_amount: 20, 
                    merchants: 30, 
                    banks: 5,
                    graph_file: String::from("/home/fanlgrp/Projects/briolette_cbdc_paper/src/simulation/briolettesim/graphs/watts-strogatz-5-connections.txt"),
                    p2m_probability: 0.2, 
                    p2p_probability: 0.6, 
                    move_probability: 0.2,  
                    random_sync_probability: 0.01,  
                    merchant_sync_frequency: 12,
                    tickets_given_right_away: 20,
                    tickets_lower_bound_to_sync: 2,
                    account_balance: 500,
                    model: Model::Rural,
                    graph_config: GraphConfig::Watts(WattsConfig {
                        size: 64, 
                        param: 15,
                    }),
                    category: ExperimentCategory::SyncParams,
                },
                ExperimentConfig { // 2:1, small top-up, merchants sync every 8 hours, 8 tickets > quick sync 
                    ratio_double_spenders_to_honest: 0.5,
                    top_up_amount: 5, 
                    merchants: 30, 
                    banks: 5,
                    graph_file: String::from("/home/fanlgrp/Projects/briolette_cbdc_paper/src/simulation/briolettesim/graphs/watts-strogatz-5-connections.txt"),
                    p2m_probability: 0.2, 
                    p2p_probability: 0.6, 
                    move_probability: 0.2,  
                    random_sync_probability: 0.01,  
                    merchant_sync_frequency: 8,
                    tickets_given_right_away: 8,
                    tickets_lower_bound_to_sync: 2,
                    account_balance: 500,
                    model: Model::Rural,
                    graph_config: GraphConfig::Watts(WattsConfig {
                        size: 64, 
                        param: 15,
                    }),
                    category: ExperimentCategory::SyncParams,
                },
            ];

            for (exp_id, exp) in predefined_experiments.into_iter().enumerate() {
                for _ in 0..repeat_experiment_number_of_times {
                    experiments
                        .entry(exp_id)
                        .or_insert_with(Vec::new)
                        .push(exp.clone());
                }
            }

            let experiments_start = Instant::now();
            println!("Starting experiments at: {:?}", experiments_start);
            run_experiments(experiments, population_size, &experiment_dir, None)?;
            let experiments_end = Instant::now();
            println!("Experiments completed in: {:?}", experiments_end.duration_since(experiments_start));
        },
        Commands::Sobol { params_file, repeat, population_size, max_threads } => {
            let mut experiments: HashMap<usize, Vec<ExperimentConfig>> = HashMap::new();
            let experiment_dir = create_experiment_directory("sobol", population_size as u64)?;
            let file = File::open(params_file)?;
            let reader: io::BufReader<File> = io::BufReader::new(file);    
            
            for (exp_param_idx, experiment_params) in reader.lines().enumerate() {
                let experiment_params: Vec<f64> = 
                    experiment_params?
                        .split_whitespace()
                        .into_iter()
                        .map(|s| {
                            s.parse::<f64>().expect("Input is not a float!")
                        })
                        .collect();
                
                if let [
                    move_probability,
                    p2p_probability,
                    p2m_probability,
                    ratio_double_spenders_to_honest
                ] = experiment_params.as_slice() {
                    for _ in 0..repeat {
                        experiments
                            .entry(exp_param_idx)
                            .or_insert_with(Vec::new)
                            .push(
                                ExperimentConfig {
                                    ratio_double_spenders_to_honest: *ratio_double_spenders_to_honest,
                                    random_sync_probability: 0.01,
                                    top_up_amount: 10,
                                    merchants: 30, 
                                    banks: 5,
                                    // graph_file: format!("/home/fanlgrp/Projects/briolette_cbdc_paper/src/simulation/briolettesim/graphs/barabasi_albert_8_m.txt"),
                                    graph_file: format!("/home/fanlgrp/Projects/briolette_cbdc_paper/src/simulation/briolettesim/graphs/watts-strogatz-5-connections.txt"),
                                    p2m_probability: *p2m_probability, 
                                    p2p_probability: *p2p_probability, 
                                    move_probability: *move_probability,   
                                    merchant_sync_frequency: 8,
                                    tickets_given_right_away: 8, 
                                    tickets_lower_bound_to_sync: 2,
                                    account_balance: 200,
                                    model: Model::Rural,
                                    graph_config: GraphConfig::Watts(WattsConfig {
                                        size: 64, 
                                        param: 5,
                                    }),
                                    // model: Model::Urban,
                                    // graph_config: GraphConfig::Barabasi(BarabasiConfig {
                                    //     size: 64, 
                                    //     param: 8,
                                    // }),
                                    category: ExperimentCategory::Sobol,
                                },
                            );
                    }
                }
            }

            let experiments_start = Instant::now();
            println!("Starting experiments at: {:?}", experiments_start);
            run_experiments(experiments, population_size as u64, &experiment_dir, max_threads)?;
            let experiments_end = Instant::now();
            println!("Experiments completed in: {:?}", experiments_end.duration_since(experiments_start));
        }
    }
    
    let program_end = Instant::now();
    let total_duration = program_end.duration_since(program_start);
    println!("Total program execution time: {:?}", total_duration);
    println!("Program completed at: {:?}", program_end);
    
    Ok(())
}

// Structure to hold experiment execution parameters
#[derive(Clone)]
struct ExperimentTask {
    experiment: ExperimentConfig,
    experiment_id: usize,
    exp_idx: usize,
    population_size: u64,
    experiment_dir: String,
}

// Function to get actual available memory from system
fn get_available_memory_mb() -> u64 {
    let meminfo = match std::fs::read_to_string("/proc/meminfo") {
        Ok(content) => content,
        Err(_) => {
            eprintln!("WARNING: Cannot read /proc/meminfo, using fallback estimate");
            return 8000; // Conservative fallback
        }
    };
    
    let mut available_kb = 0;
    for line in meminfo.lines() {
        if line.starts_with("MemAvailable:") {
            if let Some(kb_str) = line.split_whitespace().nth(1) {
                available_kb = kb_str.parse::<u64>().unwrap_or(0);
                break;
            }
        }
    }
    
    // Convert to MB
    let available_mb = available_kb / 1024;
    
    // Ensure we have at least 2GB available
    if available_mb < 2048 {
        eprintln!("WARNING: Very low memory detected ({}MB available)", available_mb);
        return 2048; // Minimum safe amount
    }
    
    println!("Detected available memory: {}MB", available_mb);
    available_mb
}

// Function to check available memory and prevent OOM
fn check_memory_usage() -> bool {
    let available_mb = get_available_memory_mb();
    let min_required_mb = 4096; // 4GB minimum for safety
    
    if available_mb < min_required_mb {
        eprintln!("WARNING: Low memory detected ({}MB available, {}MB required). Consider reducing parallel threads or population size.", 
                 available_mb, min_required_mb);
        return false;
    }
    
    println!("Memory check passed: {}MB available", available_mb);
    true
}

fn run_single_experiment(task: ExperimentTask) -> io::Result<()> {
    let individual_experiment_start = Instant::now();
    println!("Starting individual experiment {} in group {} at: {:?}", task.exp_idx, task.experiment_id, individual_experiment_start);
    
    let graph: SimulationGraph = SimulationGraph::new(task.experiment.graph_file.as_str())?;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64;
    let random_component = thread_rng().gen::<u64>();
    let unique_seed = timestamp ^ random_component;
    let mut rng: StdRng = SeedableRng::seed_from_u64(unique_seed);
    let mgr_seed = rng.gen::<u64>();

    let total_vertices_in_graph = graph.adjacency_list.len();

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

    let (num_consumers, num_double_spenders) = split_population(
        task.population_size, 
        task.experiment.ratio_double_spenders_to_honest
    ); 
    let num_merchants = task.experiment.merchants;
    let num_banks = task.experiment.banks;

    // Setup init stats
    let mut upstats = Statistics::default();

    let default_balance = task.experiment.account_balance;
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
            deposit_limit: task.experiment.top_up_amount,
            ticket_refill: task.experiment.tickets_given_right_away,
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
    for _ in 0..8 {
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
                        sync_probability: task.experiment.random_sync_probability,
                        sync_distribution: SupportedDistributions::Uniform,
                        p2m_probability: task.experiment.p2m_probability,
                        p2m_distribution: SupportedDistributions::Uniform,
                        p2p_probability: task.experiment.p2p_probability,
                        p2p_distribution: SupportedDistributions::Uniform,
                        double_spend_probability: 0.0,
                        double_spend_distribution: SupportedDistributions::Uniform,
                        max_rejections: 5,
                        move_distribution: SupportedDistributions::Uniform,
                        move_probability: task.experiment.move_probability,
                        /*(
                        step_period: 24,
                        max_transactions_per_period: 2,
                        */
                        wids: task.experiment.tickets_given_right_away,
                        wid_low_watermark: task.experiment.tickets_lower_bound_to_sync,
                        account_balance: default_balance,
                        last_requested_step: 0,
                        // (0xuki) TODO: change bank assignment procedure to account for distance to the bank!
                        bank: c % num_banks, // ensure we match the balance with the coins. TODO: Register() to get bank and balance.
                    }),
                },
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
                        wids: task.experiment.tickets_given_right_away,
                        wid_low_watermark: task.experiment.tickets_lower_bound_to_sync,
                        account_balance: default_balance,
                        last_requested_step: 0,
                        bank: rng.gen_range(0..num_banks),
                    }),
                },
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
                        wids: task.experiment.tickets_given_right_away,
                        wid_low_watermark: task.experiment.tickets_lower_bound_to_sync,
                        account_balance: default_balance,
                        last_requested_step: 0,
                        bank: rng.gen_range(0..num_banks),
                    }),
                },
            })],
            15,
        );
    }

    upstats.double_spenders_total = num_double_spenders * 2;
    upstats.total_people = upstats.double_spenders_total + num_consumers;


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
                        sync_frequency: task.experiment.merchant_sync_frequency, // every `sync_frequency` hours make a transfer to the bank -> sync as a result!
                        // sync_probability: 0.01, 
                        // sync_distribution: SupportedDistributions::Uniform,
                        account_balance: 0,
                        last_tx_step: None,
                        bank: rng.gen_range(0..num_banks),
                    }),
                },
            })],
        );
    }

    // mgr.register_observer(1, &observe);
    // mgr.register_observer(1, &check_exit_conditions_and_print_results_to_file);
    mgr.register_observer(1, check_exit_conditions_and_print_results_to_file_avged_out);
    
    match &task.experiment.graph_config {
        GraphConfig::Watts(_wc) => {

            let subdir = format!("{}/rural", task.experiment_dir);
            let file_path = format!(
                "{}/model_{:?}_p2p_{}_p2m_{}_ratiodoublespenders_{}_move_{}_expid_{}_{:?}_{}k_actors.txt",
                subdir,
                task.experiment.model,
                task.experiment.p2p_probability,
                task.experiment.p2m_probability,
                task.experiment.ratio_double_spenders_to_honest,
                task.experiment.move_probability,
                task.experiment_id,
                task.experiment.category,
                task.population_size / 1000
            );

            // Ensure the directory exists
            if let Some(parent) = Path::new(&file_path).parent() {
                fs::create_dir_all(parent)?;
            }

            mgr.run(4000, &file_path);
        },
        GraphConfig::Barabasi(_bc) => {

            let subdir = format!("{}/urban", task.experiment_dir);
            let file_path = format!(
                "{}/model_{:?}_p2p_{}_p2m_{}_ratiodoublespenders_{}_move_{}_expid_{}_{:?}_{}k_actors.txt",
                subdir,
                task.experiment.model,
                task.experiment.p2p_probability,
                task.experiment.p2m_probability,
                task.experiment.ratio_double_spenders_to_honest,
                task.experiment.move_probability,
                task.experiment_id,
                task.experiment.category,
                task.population_size / 1000
            );

            // Ensure the directory exists
            if let Some(parent) = Path::new(&file_path).parent() {
                fs::create_dir_all(parent)?;
            }

            mgr.run(4000, &file_path);
        }
    };
    
    let individual_experiment_end = Instant::now();
    println!("Individual experiment {} in group {} completed in: {:?}", task.exp_idx, task.experiment_id, individual_experiment_end.duration_since(individual_experiment_start));
    
    Ok(())
}

fn run_experiments(
    experiments: HashMap<usize, Vec<ExperimentConfig>>,
    population_size: u64,
    experiment_dir: &str,
    max_threads: Option<usize>,
) -> io::Result<()> {
    let run_experiments_start = Instant::now();
    println!("run_experiments started at: {:?}", run_experiments_start);
    
    // Check memory before starting
    if !check_memory_usage() {
        eprintln!("ERROR: Insufficient memory available. Exiting to prevent OOM.");
        process::exit(1);
    }
    
    // Create a list of all experiment tasks to be executed in parallel
    let mut experiment_tasks = Vec::new();
    
    for (experiment_id, experiment_vector) in &experiments {
        let experiment_group_start = Instant::now();
        println!("Starting experiment group {} at: {:?}", experiment_id, experiment_group_start);
        
        for (exp_idx, experiment) in experiment_vector.iter().enumerate() {
            experiment_tasks.push(ExperimentTask {
                experiment: experiment.clone(),
                experiment_id: *experiment_id,
                exp_idx,
                population_size,
                experiment_dir: experiment_dir.to_string(),
            });
        }
    }
    
    // Calculate optimal thread count based on memory and population size
    let memory_per_experiment_mb = (population_size as f64 * 0.2) as u64; // Rough estimate: 0.2MB per agent
    
    // Get actual available memory from system
    let available_memory_mb = get_available_memory_mb();
    let cpu_cores = rayon::current_num_threads();
    
    // Add 30% safety buffer to memory calculation
    let max_safe_threads = ((available_memory_mb as f64 * 0.7) as u64 / memory_per_experiment_mb).max(1).min(cpu_cores as u64) as usize;
    
    // Use user-specified threads or calculate based on memory
    let max_threads = match max_threads {
        Some(user_threads) => {
            println!("Using user-specified thread count: {}", user_threads);
            std::cmp::min(user_threads, rayon::current_num_threads())
        },
        None => {
            std::cmp::min(max_safe_threads as usize, rayon::current_num_threads())
        }
    };
    
    println!("Executing {} experiments in parallel using {} threads (memory-optimized)", 
             experiment_tasks.len(), 
             max_threads);
    println!("Estimated memory usage: {}MB per experiment, {}MB total", 
             memory_per_experiment_mb, 
             memory_per_experiment_mb * max_threads as u64);
    println!("Memory efficiency: {:.1}% of detected available memory ({}MB)", 
             (memory_per_experiment_mb * max_threads as u64) as f64 / available_memory_mb as f64 * 100.0,
             available_memory_mb);
    
    // Create a custom thread pool to avoid global pool conflicts
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(max_threads)
        .build()
        .unwrap();
    
    let results: Vec<io::Result<()>> = pool.install(|| {
        experiment_tasks
            .into_par_iter()
            .map(|task| run_single_experiment(task))
            .collect()
    });
    
    // Check for any errors in the results
    for result in results {
        result?;
    }
    
    let run_experiments_end = Instant::now();
    println!("run_experiments completed in: {:?}", run_experiments_end.duration_since(run_experiments_start));
    
    Ok(())
}

fn observe(
    step: usize, 
    world: &WorldData, 
    _pop: &SimulationPopulation<Simulator>, 
    _end_simulation_flag: &mut bool, 
    _file_path: &str   
) {
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
    if step == 72 {
        *end_simulation_flag = true;
        // Ensure the directory exists
        if let Some(parent) = Path::new(file_path).parent() {
            fs::create_dir_all(parent).unwrap_or_else(|e| {
                eprintln!("Failed to create directory {:?}: {}", parent, e);
                panic!("Failed to create directory");
            });
        }

        let mut file = OpenOptions::new()
            .create(true)  // Create the file if it doesn't exist
            .write(true)   // Open the file for writing
            .append(true)  // Append to the file if it exists
            .open(file_path)
            .unwrap_or_else(|e| {
                eprintln!("Failed to open file {}: {}", file_path, e);
                panic!("Failed to open file");
            });

        // How many simulation steps have passed since initial double-spending fork?
        for (epoch, life) in &world.statistics.double_spent_life_measurements {
            match write!(file, "{} {} ", epoch, life) {
                Ok(_) => (),
                Err(e) => panic!("{}", e) 
            }
        }
        match writeln!(file) {
            Ok(_) => (),
            Err(e) => panic!("{}", e) 
        }

        // How many transactions have passed since initial double-spending fork?
        for (epoch, txs) in &world.statistics.double_spent_txs_measurements {
            match write!(file, "{} {} ", epoch, txs) {
                Ok(_) => (),
                Err(e) => panic!("{}", e) 
            }
        }
        match writeln!(file) {
            Ok(_) => (),
            Err(e) => panic!("{}", e) 
        }

        // Step when all cheaters have been caught
        match writeln!(file, "{}", step) {
            Ok(_) => (),
            Err(e) => panic!("{}", e) 
        }

        // Ratio (double_spent_coins / coins_total)
        for item in &world.statistics.ratios_of_double_spent_coins {
            match write!(file, "{} ", item) {
                Ok(_) => (),
                Err(e) => panic!("{}", e) 
            }
        }
        match writeln!(file) {
            Ok(_) => (),
            Err(e) => panic!("{}", e) 
        }

        // Ratio (double_spenters / double_spenders_total) (???)
        for item in &world.statistics.ratio_of_double_spenders_caught {
            match write!(file, "{} ", item) {
                Ok(_) => (),
                Err(e) => panic!("{}", e) 
            }
        }
        match writeln!(file) {
            Ok(_) => (),
            Err(e) => panic!("{}", e) 
        }

        // mean/std/max for (global - local) epoch diffs
        for item in &world.statistics.global_to_local_epoch_diffs {
            match write!(file, "{} {} {}:", item.mean, item.standard_deviation, item.max_diff) {
                Ok(_) => (),
                Err(e) => panic!("{}", e) 
            }
        }
        match writeln!(file) {
            Ok(_) => (),
            Err(e) => panic!("{}", e) 
        }

        // std for intra-sample epoch diffs
        for item in &world.statistics.std_local_epoch_diffs {
            match write!(file, "{} ", item) {
                Ok(_) => (),
                Err(e) => panic!("{}", e) 
            }
        }
    }
}



fn check_exit_conditions_and_print_results_to_file_avged_out(
    step: usize, 
    world: &WorldData, 
    _pop: &SimulationPopulation<Simulator>,
    end_simulation_flag: &mut bool, 
    file_path: &str
) {
    if step == 72 {
        *end_simulation_flag = true;
        // Ensure the directory exists
        if let Some(parent) = Path::new(file_path).parent() {
            fs::create_dir_all(parent).unwrap_or_else(|e| {
                eprintln!("Failed to create directory {:?}: {}", parent, e);
                panic!("Failed to create directory");
            });
        }

        let mut file = OpenOptions::new()
            .create(true)  // Create the file if it doesn't exist
            .write(true)   // Open the file for writing
            .append(true)  // Append to the file if it exists
            .open(file_path)
            .unwrap_or_else(|e| {
                eprintln!("Failed to open file {}: {}", file_path, e);
                panic!("Failed to open file");
            });

        // Ratio (double_spent_coins / coins_total)
        for item in &world.statistics.ratios_of_double_spent_coins {
            match write!(file, "{} ", item) {
                Ok(_) => (),
                Err(e) => panic!("{}", e) 
            }
        }
        match writeln!(file) {
            Ok(_) => (),
            Err(e) => panic!("{}", e) 
        }

        // Ratio (double_spenters / double_spenders_total) (???)
        for item in &world.statistics.ratio_of_double_spenders_caught {
            match write!(file, "{} ", item) {
                Ok(_) => (),
                Err(e) => panic!("{}", e) 
            }
        }
        match writeln!(file) {
            Ok(_) => (),
            Err(e) => panic!("{}", e) 
        }

        // mean/std/max for (global - local) epoch diffs
        for item in &world.statistics.global_to_local_epoch_diffs {
            match write!(file, "{} {} {}:", item.mean, item.standard_deviation, item.max_diff) {
                Ok(_) => (),
                Err(e) => panic!("{}", e) 
            }
        }
        match writeln!(file) {
            Ok(_) => (),
            Err(e) => panic!("{}", e) 
        }
    }
}