package tsml.classifiers.distance_based.utils.scoring;

import utilities.Utilities;
import weka.core.Instances;

import java.util.List;

/**
 * Purpose: score the partitioning of some data into subsets.
 */
public interface Scorer {
    double findScore(Instances parent, List<Instances> children);

    class GiniImpurity implements Scorer {

        @Override
        public double findScore(final Instances parent, final List<Instances> children) {
            return ScoreUtils.giniImpurity(parent, children);
        }
    }

    class InfoGain implements Scorer {

        @Override
        public double findScore(final Instances parent, final List<Instances> children) {
            return ScoreUtils.infoGain(parent, children);
        }
    }

    class GiniImpurityEntropy implements Scorer {

        @Override
        public double findScore(final Instances parent, final List<Instances> children) {
            return ScoreUtils.giniImpurityEntropy(children);
        }
    }

    class InfoGainEntropy implements Scorer {

        @Override
        public double findScore(final Instances parent, final List<Instances> children) {
            return ScoreUtils.infoGainEntropy(children);
        }
    }
}
