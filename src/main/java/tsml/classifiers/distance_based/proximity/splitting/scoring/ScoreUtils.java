package tsml.classifiers.distance_based.proximity.splitting.scoring;

import java.util.List;
import utilities.Utilities;
import weka.core.Instances;

/**
 * Purpose: utils class for scorers.
 * <p>
 * Contributors: goastler
 */
public class ScoreUtils {

    private ScoreUtils() {

    }

    private static Scorer GINI_IMPURITY;

    public static class GiniImpurity implements Scorer {

        @Override
        public double findScore(final Instances parent, final List<Instances> parts) {
            return Utilities.giniImpurity(parent, parts);
        }
    }

    public static Scorer getGlobalGiniImpurityScorer() {
        if(GINI_IMPURITY == null) {
            GINI_IMPURITY = new GiniImpurity();
        }
        return GINI_IMPURITY;
    }

    private static Scorer INFO_GAIN;

    public static Scorer getGlobalInfoGainScorer() {
        if(INFO_GAIN == null) {
            INFO_GAIN = new InfoGain();
        }
        return INFO_GAIN;
    }

    public static class InfoGain implements Scorer {
        @Override
        public double findScore(final Instances parent, final List<Instances> parts) {
            return Utilities.infoGain(parent, parts);
        }
    }
}
