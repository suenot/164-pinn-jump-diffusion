# PINN for Jump Diffusion -- Explained Simply

## The Big Idea in One Sentence

We teach an AI to price financial bets (options) in a world where prices can suddenly jump -- like how earthquakes can shake a city without warning -- using physics rules baked into the AI's learning process.

---

## Part 1: Why Do We Need This?

### The Weather Forecast Analogy

Imagine you are a weather forecaster. Your basic model predicts temperature changes smoothly: it gets a little warmer each hour during the day, a little cooler at night. This works fine most of the time.

But sometimes a sudden storm blows in. The temperature drops 15 degrees in 10 minutes. Your smooth model never predicted this because **it doesn't know about sudden storms**.

This is exactly what happens in financial markets:

- **Normal days**: Prices wiggle up and down a tiny bit -- like gentle temperature changes
- **Jump days**: A company announces bankruptcy, or a crypto exchange gets hacked, and the price drops 20% in seconds -- like a sudden storm

The old model (Black-Scholes, from 1973) only handles the gentle wiggles. It is **blind to sudden jumps**.

### Real Examples of "Jumps"

- **Bitcoin, March 2020**: Dropped from $8,000 to $3,800 in one day (COVID panic)
- **Bitcoin, May 2021**: Crashed 30% when China banned crypto mining
- **FTX collapse, Nov 2022**: Multiple coins lost 50-90% overnight
- **Stocks**: Flash crashes, earnings surprises, merger announcements

---

## Part 2: Merton's Jump Diffusion -- Adding Storms to the Forecast

In 1976, economist Robert Merton had a brilliant idea: **what if we add random jumps to our price model?**

Think of it like this:

```
Price movement = Normal wiggles + Occasional sudden jumps
```

The "wiggles" happen every second, like wind blowing leaves around. The "jumps" happen randomly -- maybe 5-10 times per year for Bitcoin -- like lightning strikes.

Each jump has:
- **How often?** (lambda) -- On average, how many jumps per year?
- **How big?** (mu_J) -- Is the average jump up or down, and by how much?
- **How variable?** (sigma_J) -- Are all jumps similar size, or can they range from tiny to enormous?

### The Problem: A Harder Equation

When we add jumps, the math equation for pricing options becomes much harder. Instead of a regular equation (like `2x + 3 = 7`), we get an **integro-differential equation** -- it has both derivatives (rates of change) AND an integral (a sum over all possible jump sizes).

It is like asking: "What is the option worth if the price could jump by ANY amount, with some jumps more likely than others?"

That integral -- the "sum over all possible jumps" -- is what makes this so tricky.

---

## Part 3: What is a PINN?

### The Tutor Analogy

Imagine teaching a student to solve physics problems. You could:

1. **Method A (Traditional)**: Give them a formula sheet and have them plug in numbers. Fast for simple problems, but breaks for complex ones.

2. **Method B (PINN)**: Have the student guess answers, but every time they guess, you check if their answer **obeys the laws of physics**. If not, you say "wrong, try again." Over time, they learn to always give answers that respect physics.

A PINN (Physics-Informed Neural Network) is Method B:
- It is a neural network (an AI that can approximate any function)
- It is trained so that its outputs **satisfy the physics equation** (our jump diffusion equation)
- It learns the laws of physics, not just data patterns

### Why is This Better?

- **Knows physics**: Even without much data, it produces reasonable answers because it respects the equation
- **Fast after training**: Once trained, it gives answers in microseconds
- **Automatic sensitivities**: It can instantly tell you how the answer changes if you tweak any input (these are called "Greeks" in finance)

---

## Part 4: Handling Jumps in the PINN

### The Restaurant Analogy

Imagine you own a restaurant and want to predict tomorrow's revenue.

**Without jumps**: Revenue changes smoothly based on day of week, weather, etc.

**With jumps**: Sometimes a tour bus of 50 people shows up unexpectedly! To account for this, you need to consider: "What would my revenue be if 10 extra people showed up? 20? 50? 100?" and weight each scenario by its probability.

This "considering all possible surprises" is the integral in our equation. For our option pricing PINN:

1. We pick a spot price, say $50,000 for Bitcoin
2. We ask: "What if the price jumps to $45,000? $40,000? $55,000? $60,000?"
3. For EACH possible jumped price, we evaluate what the option would be worth
4. We average all these values, weighted by how likely each jump size is
5. This average is the "integral term"

### Gauss-Hermite Quadrature: A Smart Shortcut

Computing an integral over infinitely many possible jump sizes sounds impossible. But there is a trick: **Gauss-Hermite quadrature**.

Think of it like a survey. To know the average height of all humans, you don't measure everyone -- you pick a smart sample of ~30 people at carefully chosen heights and weight their answers. The math guarantees this gives a very accurate average.

Similarly, instead of checking infinitely many jump sizes, we check about 32 carefully chosen ones. The mathematical theory ensures this gives us excellent accuracy.

---

## Part 5: Putting It All Together

### The Training Process

```
Step 1: Generate random (price, time) pairs in our domain
Step 2: For each pair, the neural network guesses the option value
Step 3: Check if the guess obeys our physics equation:
        - Do the derivatives add up correctly? (PDE part)
        - Is the integral over jumps consistent? (integral part)
        - Does it match the known payoff at expiration? (boundary condition)
Step 4: Compute how wrong the guess is (the "loss")
Step 5: Adjust the network to reduce the wrongness
Step 6: Repeat 10,000-50,000 times
```

After training:
- The network can price ANY option with these parameters in ~0.1 milliseconds
- It automatically knows the sensitivities (Greeks)
- It naturally produces the "volatility smile" that markets show

### The Volatility Smile

If you use the PINN's option prices and convert them back to Black-Scholes implied volatility, you see a U-shaped curve across strikes -- the famous **volatility smile**. This happens because:

- Options far from the current price (deep out-of-the-money) are worth more than Black-Scholes predicts
- This is because jumps make extreme outcomes more likely
- The smile is the market's way of saying "we know jumps happen"

---

## Part 6: Why Crypto Markets?

Crypto markets on exchanges like Bybit are the perfect testing ground because:

1. **More jumps**: Bitcoin has roughly 10-15 "jump days" per year vs 2-5 for typical stocks
2. **Bigger jumps**: 10-30% daily moves happen regularly in crypto
3. **24/7 trading**: No overnight gaps -- jumps happen in real-time
4. **Active options markets**: Bybit and Deribit have liquid BTC/ETH options
5. **Clear smile**: The volatility smile is very pronounced in crypto options

---

## Part 7: The Trading Strategy

The strategy is simple in concept:

1. Use our PINN to calculate the "fair" price of an option (accounting for jumps)
2. Compare with the market price
3. If the market price is too low (market underestimates jump risk) -- **buy the option**
4. If the market price is too high -- **sell the option**
5. Hedge the directional risk (delta-hedge) so we only profit from the jump mispricing

It is like finding a store that sells umbrellas too cheaply before storm season -- you buy them because you know storms are coming.

---

## Quick Summary

| Concept | Simple Explanation |
|---------|-------------------|
| Jump Diffusion | Price model that includes both normal wiggles AND sudden jumps |
| PIDE | The math equation that jump diffusion options must satisfy |
| PINN | AI that learns to solve the equation by being penalized for physics violations |
| Gauss-Hermite | Smart trick to efficiently compute the "sum over all possible jumps" |
| Volatility Smile | The U-shaped pattern of implied volatility that jumps naturally produce |
| Greeks | Sensitivities of option price to various inputs, computed for free by the PINN |
| Bybit | Crypto exchange where jump behavior is especially prominent |

---

## One-Paragraph Summary

Standard option pricing models assume prices change smoothly, but real markets -- especially crypto -- have sudden jumps (crashes, rallies). Merton's Jump Diffusion model accounts for these jumps, but its pricing equation includes a tricky integral over all possible jump sizes. We use a Physics-Informed Neural Network (PINN) to solve this equation: the AI learns to output option prices that obey the jump-diffusion physics, handling the integral via a clever sampling technique (Gauss-Hermite quadrature). The result is a fast, accurate pricing engine that naturally produces realistic volatility smiles and Greeks, making it ideal for trading crypto options on exchanges like Bybit.
