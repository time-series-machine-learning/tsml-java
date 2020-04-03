package tsml.classifiers.distance_based.proximity.splitting;

import utilities.Utilities;
import weka.core.Instances;

import java.util.List;

public abstract class Scorer {
    public abstract double findScore(Instances parent, List<Instances> parts);

    private static final Scorer GINI_IMPURITY = new Scorer() {
        @Override
        public double findScore(final Instances parent, final List<Instances> parts) {
            return Utilities.giniImpurity(parent, parts);
        }
    };

    public static Scorer getGiniImpurityScorer() {
        return GINI_IMPURITY;
    }

    private static final Scorer INFO_GAIN = new Scorer() {
        @Override
        public double findScore(final Instances parent, final List<Instances> parts) {
            return Utilities.infoGain(parent, parts);
        }
    };

    public static Scorer getInfoGainScorer() {
        return INFO_GAIN;
    }


}
