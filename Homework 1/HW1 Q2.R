# EE5393 Homework 1 Q2
# Kenji Vang | vang3841@umn.edu
# ChatGPT has helped generate some code in this file
# HW1 Q2.R required both lambda.in and lambda.r to exist in the same folder to execute correctly.

# ----------------------------
# Parameters
MOI_range <- 1:10
n_sim <- 1000      # stochastic runs per MOI
max_steps <- 10000 # maximum reactions per simulation

# Thresholds
stealth_thresh <- 145
hijack_thresh <- 55

# ----------------------------
# Initialize results to ensure it exists
results <- data.frame(MOI=integer(), P_stealth=double(), P_hijack=double())

# ----------------------------
# Load reaction and initial value files
lambda_file <- "lambda.r"
init_file   <- "lambda.in"

lambda_lines <- readLines(lambda_file)
init_lines   <- readLines(init_file)

# ----------------------------
# Parse initial values
init_vals <- sapply(init_lines, function(line) as.numeric(strsplit(line, " ")[[1]][2]))
names(init_vals) <- sapply(init_lines, function(line) strsplit(line, " ")[[1]][1])
init_vals <- as.list(init_vals)

# Ensure essential species exist
if (is.null(init_vals$cI2)) init_vals$cI2 <- 0
if (is.null(init_vals$Cro2)) init_vals$Cro2 <- 0
if (is.null(init_vals$MOI)) init_vals$MOI <- 1

# ----------------------------
# Parse reactions
reactions <- lapply(lambda_lines, function(line) {
  parts <- strsplit(line, ":")[[1]]
  reactants <- strsplit(trimws(parts[1]), " ")[[1]]
  products  <- strsplit(trimws(parts[2]), " ")[[1]]
  rate      <- as.numeric(trimws(parts[3]))
  list(reactants=reactants, products=products, rate=rate)
})

# ----------------------------
# Gillespie SSA function
run_gillespie <- function(state, reactions, max_steps, stealth_thresh, hijack_thresh) {
  
  # First check before any reactions
  if (!is.null(state$cI2) && state$cI2 > stealth_thresh) return("stealth")
  if (!is.null(state$Cro2) && state$Cro2 > hijack_thresh) return("hijack")
  
  for (step in 1:max_steps) {
    # Compute propensities
    propensities <- sapply(reactions, function(r) {
      counts <- sapply(r$reactants, function(s) ifelse(!is.null(state[[s]]), state[[s]], 0))
      if (any(counts <= 0)) return(0)
      r$rate * prod(counts)
    })
    
    a0 <- sum(propensities)
    if (a0 == 0) break
    
    # Select reaction
    r_index <- sample(length(reactions), 1, prob=propensities)
    r <- reactions[[r_index]]
    
    # Update state counts
    for (s in r$reactants) state[[s]] <- state[[s]] - 1
    for (s in r$products) {
      if (is.null(state[[s]])) state[[s]] <- 0
      state[[s]] <- state[[s]] + 1
    }
    
    # Check thresholds
    if (!is.null(state$cI2) && state$cI2 > stealth_thresh) return("stealth")
    if (!is.null(state$Cro2) && state$Cro2 > hijack_thresh) return("hijack")
  }
  
  return("none")
}

# ----------------------------
# Main loop over MOI
for (MOI in MOI_range) {
  stealth_count <- 0
  cat("Running MOI =", MOI, "\n")  # progress indicator
  
  for (sim in 1:n_sim) {
    state <- init_vals
    state$MOI <- MOI
    
    outcome <- run_gillespie(state, reactions, max_steps, stealth_thresh, hijack_thresh)
    if (outcome == "stealth") stealth_count <- stealth_count + 1
  }
  
  P_stealth <- stealth_count / n_sim
  P_hijack <- 1 - P_stealth
  results <- rbind(results, data.frame(MOI=MOI, P_stealth=P_stealth, P_hijack=P_hijack))
}

# ----------------------------
# Print final results
cat("\nSimulation complete!\n")
print(results)
