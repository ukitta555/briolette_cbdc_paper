pub fn split_population(
    total_agents_in_population: u64,
    bad_to_good_agents_ratio: f64, 
) -> (usize, usize) {
    let bad_actors: usize = (
        (total_agents_in_population as f64) * 
        ((bad_to_good_agents_ratio / 2.0) / (1.0 + bad_to_good_agents_ratio))
    ).floor() as usize;
    let good_actors: usize = (
        (total_agents_in_population as f64) * 
        (1.0 / (1.0 + bad_to_good_agents_ratio))
    ).floor() as usize;

    (good_actors, bad_actors) 
}