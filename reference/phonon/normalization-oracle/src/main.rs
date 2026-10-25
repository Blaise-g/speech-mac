#![allow(dead_code)]

use std::io::{self, BufRead};

#[derive(Debug)]
pub enum Error { InvalidArgument(String) }
pub type Result<T> = std::result::Result<T, Error>;
impl Error { pub fn invalid_argument(message: String) -> Self { Self::InvalidArgument(message) } }
mod preprocess;

fn main() {
    for line in io::stdin().lock().lines() {
        let line = line.expect("read input");
        println!("{}", preprocess::normalize_text(&line, preprocess::Lang::En, preprocess::Rules::DEFAULT));
    }
}
