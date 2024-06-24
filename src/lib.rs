pub mod anomaly;
pub mod common;
pub mod datasets;
pub mod metrics;
pub mod mondrian_forest;
pub mod stream;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
