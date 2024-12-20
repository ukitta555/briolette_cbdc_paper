use std::{fs::File, io::{self, BufRead}};

use serde::{Deserialize, Serialize};



#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Copy, Default)]
pub struct GraphVertexIndex(pub usize);

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Default)]
pub struct Vertex {
    idx: usize,
    pub agents: Vec<usize>,
    pub resources: Vec<usize>,
}

impl Vertex {
    pub fn new(idx: usize) -> Self {
        Self {
            idx,
            agents: Vec::new(),
            resources: Vec::new(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Default)]
pub struct SimulationGraph {
    pub adjacency_list: Vec<Vec<usize>>,
    pub vertices: Vec<Vertex>
}

impl SimulationGraph {
    pub fn new(path: &str) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let mut reader = io::BufReader::new(file);
        
        let number_of_vertices = reader.lines().count();
        println!("Number of vertices in a graph that is being read: {}", number_of_vertices);
        let mut graph: Vec<Vec<usize>> = Vec::new(); 
    
        file = File::open(path)?;
        reader = io::BufReader::new(file);
    
        for neighbours_serialized in reader.lines() {
            let neighbours: Vec<usize> = 
                neighbours_serialized?
                    .split(' ')
                    .into_iter()
                    .skip(1) // skip the idx of the node since it is irrelevant
                    .map(|s| { 
                        s.parse::<usize>().expect("Input is not an integer!")
                    })
                    .collect();
    
            graph.push(neighbours);
        }
    
        for (idx, node_neighbours) in (&graph).into_iter().enumerate() {
            print!("Node #{idx} neighbours: ");
    
            for neighbour in node_neighbours {
                print!("{neighbour} ");
            }   
            println!();
        }
    
        let mut vertices = Vec::new();
        for idx in 0..graph.len() {
            vertices.push(Vertex::new(idx));
        }

        Ok(Self {
            adjacency_list: graph,
            vertices,
        })      
    }

    pub fn reset(&mut self) {
        self.vertices = Vec::new();
        for idx in 0..self.adjacency_list.len() {
            self.vertices.push(Vertex::new(idx));
        }
    }

    pub fn at_location_mut(&mut self, idx: &GraphVertexIndex) -> &mut Vertex {
        &mut self.vertices[idx.0]     
    }
    
    pub fn at_location(&self, idx: &GraphVertexIndex) -> &Vertex {
        &self.vertices[idx.0]
    }

    pub fn neighbours(&self, idx: &GraphVertexIndex) -> &Vec<usize> {
        &self.adjacency_list[idx.0]
    }

    pub fn get_index(&self, idx: GraphVertexIndex) -> usize {
        idx.0
    }
    pub fn get_location(&self, idx: usize) -> usize {
        idx
    }
}





#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct GridCell {
    agents: Vec<usize>,
    resources: Vec<usize>,
}

impl GridCell {
    pub fn new() -> GridCell {
        GridCell {
            agents: Vec::new(),
            resources: Vec::new(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct Grid<T> {
    bounds: (usize, usize),
    cells: Vec<T>,
}

impl Grid<GridCell> {
    pub fn new(bounds: (usize, usize)) -> Grid<GridCell> {
        Grid {
            bounds: bounds,
            cells: vec![GridCell::new(); bounds.0 * bounds.1],
        }
    }
    pub fn reset(&mut self) {
        self.cells = vec![GridCell::new(); self.bounds.0 * self.bounds.1];
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Copy, Default)]
pub struct Location(usize, usize);


pub trait GridIndex<U> {
    fn at_location_mut(&mut self, l: Location) -> &mut U;
    fn at_location(&self, l: Location) -> &U;

    fn get_index(&self, l: Location) -> usize;
    fn get_location<T: Into<usize>>(&self, index: T) -> Location;
}



impl<U> GridIndex<U> for Grid<U> {
    fn at_location_mut(&mut self, l: Location) -> &mut U {
        let i = self.get_index(l);
        &mut self.cells[i]
    }
    fn at_location(&self, l: Location) -> &U {
        let i = self.get_index(l);
        &self.cells[i]
    }
    fn get_index(&self, l: Location) -> usize {
        let Location(x_u64, y_u64) = l;
        let x: usize = x_u64.try_into().unwrap();
        let y: usize = y_u64.try_into().unwrap();
        let w: usize = self.bounds.0.try_into().unwrap();
        w * y + x
    }
    fn get_location<T: Into<usize>>(&self, index: T) -> Location {
        let w: usize = self.bounds.0.try_into().unwrap();
        let i: usize = index.try_into().unwrap();
        Location(
            (i % w).try_into().unwrap(),
            ((i - (i % w)) / w).try_into().unwrap(),
        )
    }
}
