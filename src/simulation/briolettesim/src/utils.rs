pub fn split_population(
    total_agents_in_population: u64,
    bad_agents_to_population_ratio: f64, 
) -> (usize, usize) {
    let bad_actors: usize = 1.max((
        (total_agents_in_population as f64) * 
        (bad_agents_to_population_ratio / 2.0)
    ).floor() as usize);
    let good_actors: usize = total_agents_in_population as usize - 2 * bad_actors;

    (good_actors, bad_actors) 
}