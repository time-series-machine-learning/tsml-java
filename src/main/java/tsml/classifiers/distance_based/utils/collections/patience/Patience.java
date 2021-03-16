package tsml.classifiers.distance_based.optimised;

import tsml.classifiers.distance_based.utils.collections.checks.Checks;

public class Patience {
    
    public Patience() {
        setMaximise();
        setWindowSize(5);
        setTolerance(0);
        reset();
        expired = false;
    }
    
    public Patience(int windowSize) {
        this();
        setWindowSize(windowSize);
    }
    
    private int index;
    private int windowSize;
    private double best;
    private int bestIndex;
    private double tolerance;
    private boolean minimise;
    private int windowStart;
    private boolean expired;

//    /**
//     * Has the score improved within the window?
//     * @return
//     */
//    public boolean hasImproved() {
//        return index - windowIndex < windowSize;
//    }
    
    private boolean isBetter(double value) {
        if(minimise) {
            return value < best - tolerance;
        } else {
            return value > best + tolerance;
        }
    }
    
    public boolean add(double value) {
        index++;
        boolean better = isBetter(value); 
        if(better) {
            best = value;
            bestIndex = index;
            windowStart = bestIndex;
        }
        expired = (index - windowStart) >= windowSize;
        return better;
    }
    
    public void reset() {
        index = -1;
        best = -1;
        bestIndex = -1;
        windowStart = 0;
        expired = false;
    }

    public int getWindowStart() {
        return windowStart;
    }
    
    public void resetPatience() {
        windowStart = index;
    }

    public boolean isExpired() {
        return expired;
    }
    
    public int getIndex() {
        return index;
    }

    public int getBestIndex() {
        return bestIndex;
    }

    public double getBest() {
        return best;
    }

    public double getTolerance() {
        return tolerance;
    }

    public void setTolerance(final double tolerance) {
        this.tolerance = Checks.requireNonNegative(tolerance);
    }

    public int getWindowSize() {
        return windowSize;
    }

    public void setWindowSize(final int windowSize) {
        this.windowSize = Checks.requirePositive(windowSize);
    }

    public boolean isMinimise() {
        return minimise;
    }

    public void setMinimise() {
        this.minimise = true;
    }
    
    public boolean isMaximise() {
        return !isMinimise();
    }
    
    public void setMaximise() {
        minimise = false;
    }
    
    public int size() {
        return index + 1;
    }
    
    public boolean isEmpty() {
        return size() == 0;
    }
}
