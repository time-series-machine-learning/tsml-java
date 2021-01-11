package tsml.classifiers.distance_based.utils.stats.scoring;

import java.util.Objects;

import static tsml.classifiers.distance_based.utils.collections.checks.Checks.assertReal;

public class Label<A> {
    private final A id;
    private final double weight;

    public Label(A id, double weight) {
        this.weight = weight;
        this.id = Objects.requireNonNull(id);
        assertReal(weight);
    }
    
    public Label(A id) {
        this(id, 1d);
    }
    
    public double getWeight() {
        return weight;
    }

    public A getId() {
        return id;
    }

    @Override public String toString() {
        return id + ": " + weight;
    }
}
