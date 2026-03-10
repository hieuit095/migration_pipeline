use anyhow::Result;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;
use dotenv::dotenv;

mod config;
mod agents;
mod skills;
mod pipeline;
mod utils;

#[tokio::main]
async fn main() -> Result<()> {
    // Load environment variables
    dotenv().ok();

    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)
        .expect("setting default subscriber failed");

    info!("Starting AI-Powered Legacy Code Migration Pipeline");

    // TODO: Initialize Agents
    // TODO: Start Pipeline Loop

    Ok(())
}
