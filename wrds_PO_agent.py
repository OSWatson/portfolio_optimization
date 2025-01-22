import wrds
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from langchain_openai import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# Step 1: Connect to WRDS
def connect_to_wrds():
    try:
        print("Connecting to WRDS...")
        conn = wrds.Connection()
        print("Connected successfully!")
        return conn
    except Exception as e:
        print(f"Error connecting to WRDS: {e}")
        return None

# Step 2: Get PERMNOs by Tickers
def get_permnos_by_tickers(conn, tickers):
    try:
        print(f"Fetching PERMNOs for tickers: {', '.join(tickers)}...")
        query = f"""
            SELECT DISTINCT permno, ticker, comnam
            FROM crsp.stocknames
            WHERE ticker IN ({','.join([f"'{ticker}'" for ticker in tickers])})
        """
        result = conn.raw_sql(query)
        if result.empty:
            print("No matching PERMNOs found for the given tickers.")
            return pd.DataFrame()  # Return empty DataFrame
        
        deduplicated = result.drop_duplicates(subset=['ticker'], keep='last')
        print(f"Filtered PERMNOs:\n{deduplicated}")
        return deduplicated
    except Exception as e:
        print(f"Error fetching PERMNOs: {e}")
        return pd.DataFrame()

# Step 3: Query WRDS for Financial Data
def query_financial_data(conn, start_date, end_date, permnos):
    try:
        print(f"Querying WRDS for data from {start_date} to {end_date} for PERMNOs: {permnos}...")
        query = f"""
            SELECT permno, date, prc AS price, vol AS volume, ret AS return
            FROM crsp.dsf
            WHERE date BETWEEN '{start_date}' AND '{end_date}'
            AND permno IN ({','.join(map(str, permnos))})
        """
        df = conn.raw_sql(query)
        print(f"Data retrieved successfully! {len(df)} rows fetched.")
        return df
    except Exception as e:
        print(f"Error querying data: {e}")
        return None

# Step 4: Pre-compute Portfolio Metrics
def compute_portfolio_metrics(df, permnos_data):
    df['date'] = pd.to_datetime(df['date'])
    returns_df = df.pivot(index='date', columns='permno', values='return').dropna()

    permno_to_ticker = dict(zip(permnos_data['permno'], permnos_data['ticker']))
    valid_permnos = [permno for permno in returns_df.columns if permno in permno_to_ticker]
    valid_tickers = [permno_to_ticker[permno] for permno in valid_permnos]

    returns_df = returns_df[valid_permnos]
    returns_df.columns = valid_tickers

    expected_returns = returns_df.mean()
    cov_matrix = returns_df.cov()

    return returns_df, expected_returns, cov_matrix

# Step 5: Optimize Portfolio for Target Risk
def portfolio_for_target_risk(expected_returns, cov_matrix, target_risk):
    num_assets = len(expected_returns)

    def risk_difference(weights):
        # Calculate portfolio risk and find the difference from target risk
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return abs(portfolio_std_dev - target_risk)

    # Constraint 1: Weights must sum to 1
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    
    # Bounds: Minimum 5%, Maximum 30% for each asset
    bounds = [(0.05, 0.30) for _ in range(num_assets)]
    
    # Initial guess: Evenly distributed weights
    initial_weights = np.ones(num_assets) / num_assets

    # Perform optimization
    result = minimize(risk_difference, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        return result.x
    else:
        raise ValueError("Optimization failed to find a portfolio with the desired risk level.")

# Step 6: Additional Plotting Functions
def plot_sharpe_ratio_distribution(expected_returns, cov_matrix, num_portfolios=10000, risk_free_rate=0.02):
    num_assets = len(expected_returns)
    sharpe_ratios = []

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        portfolio_return = np.dot(weights, expected_returns)
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev

        sharpe_ratios.append(sharpe_ratio)

    plt.hist(sharpe_ratios, bins=50, color='blue', alpha=0.7)
    plt.title('Sharpe Ratio Distribution')
    plt.xlabel('Sharpe Ratio')
    plt.ylabel('Frequency')
    plt.show()

def plot_risk_vs_return(expected_returns, cov_matrix):
    risk = np.sqrt(np.diag(cov_matrix))
    plt.scatter(risk, expected_returns, color='purple', alpha=0.7)
    for i, ticker in enumerate(expected_returns.index):
        plt.text(risk[i], expected_returns[i], ticker, fontsize=9)
    plt.title('Risk vs Return')
    plt.xlabel('Risk (Standard Deviation)')
    plt.ylabel('Expected Return')
    plt.grid()
    plt.show()

def plot_portfolio_allocation(weights, tickers):
    plt.bar(tickers, weights, color='teal')
    plt.title('Portfolio Allocation')
    plt.xlabel('Assets')
    plt.ylabel('Weight')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_correlation_heatmap(returns_df):
    correlation_matrix = returns_df.corr()
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
    plt.colorbar(label='Correlation Coefficient')
    plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation=45, fontsize=9)
    plt.yticks(range(len(correlation_matrix)), correlation_matrix.columns, fontsize=9)
    plt.title('Correlation Heatmap')
    plt.grid(False)
    plt.show()

# Step 7: Update the Agent with Plotting Tools
def create_agent(df, returns_df, expected_returns, cov_matrix):
    try:
        print("Creating LangChain agent...")
        llm = OpenAI(temperature=0, openai_api_key="")  # Replace with your actual API key

        def optimized_weights():
            weights = optimize_portfolio(expected_returns, cov_matrix)
            return f"Optimized Weights: {dict(zip(returns_df.columns, weights))}"

        def target_risk_portfolio():
            target_risk = float(input("Enter your desired risk (standard deviation, e.g., 0.02): "))
            weights = portfolio_for_target_risk(expected_returns, cov_matrix, target_risk)
            return f"Portfolio weights for risk {target_risk}: {dict(zip(returns_df.columns, weights))}"

        def plot_frontier():
            plot_efficient_frontier(expected_returns, cov_matrix)
            return "Efficient frontier plotted successfully."

        def show_sharpe_ratio_distribution():
            plot_sharpe_ratio_distribution(expected_returns, cov_matrix)
            return "Sharpe ratio distribution plotted successfully."

        def show_risk_vs_return():
            plot_risk_vs_return(expected_returns, cov_matrix)
            return "Risk vs Return graph plotted successfully."

        def show_portfolio_allocation():
            weights = [0.1] * len(expected_returns)  # Example weights
            plot_portfolio_allocation(weights, returns_df.columns)
            return "Portfolio allocation plotted successfully."

        def show_correlation_heatmap():
            plot_correlation_heatmap(returns_df)
            return "Correlation heatmap plotted successfully."

        custom_tools = {
            "Optimized Weights": optimized_weights,
            "Show Efficient Frontier": plot_frontier,
            "Target Risk Portfolio": target_risk_portfolio,
            "Show Sharpe Ratio Distribution": show_sharpe_ratio_distribution,
            "Show Risk vs Return": show_risk_vs_return,
            "Show Portfolio Allocation": show_portfolio_allocation,
            "Show Correlation Heatmap": show_correlation_heatmap,
        }

        agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
        print("Agent created successfully!")
        return agent, custom_tools
    except Exception as e:
        print(f"Error creating agent: {e}")
        return None, {}

# Step 8: User Interaction Loop
def run_agent(agent, custom_tools):
    print("\nLangChain Agent is ready! Ask your questions about portfolio optimization.")
    print("Type 'exit' to quit.\n")
    while True:
        query = input("Your question: ")
        if query.lower() == "exit":
            print("Exiting... Goodbye!")
            break

        if query in custom_tools:
            try:
                response = custom_tools[query]()
            except Exception as e:
                response = f"Error executing custom tool: {e}"
        else:
            try:
                response = agent.run(query)
            except Exception as e:
                response = f"Error processing query: {e}"
        
        print("\nAgent Response:\n", response)

# Main Program
if __name__ == "__main__":
    conn = connect_to_wrds()
    if not conn:
        exit()

    tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "META", "NVDA", "BRK.B", "JPM", "JNJ"]
    start_date = "2020-01-01"
    end_date = "2025-01-01"

    permnos_data = get_permnos_by_tickers(conn, tickers)
    if permnos_data.empty:
        print("No valid PERMNOs found. Exiting...")
        exit()

    permnos = permnos_data['permno'].tolist()
    df = query_financial_data(conn, start_date, end_date, permnos)
    if df is None or df.empty:
        print("No data fetched. Exiting...")
        exit()

    returns_df, expected_returns, cov_matrix = compute_portfolio_metrics(df, permnos_data)

    agent, custom_tools = create_agent(df, returns_df, expected_returns, cov_matrix)
    if not agent:
        exit()

    run_agent(agent, custom_tools)
