package tsml.classifiers.distance_based.utils.stats.scoring;

import weka.core.Instances;

import java.io.Serializable;
import java.util.List;

/**
 * Purpose: score the partitioning of some data into subsets.
 */
public interface PartitionScorer extends Serializable {
    double findScore(Instances parent, List<Instances> children);

    class GiniGain implements PartitionScorer {

        @Override
        public double findScore(final Instances parent, final List<Instances> children) {
            return ScoreUtils.giniGain(parent, children);
        }
    }

    class InfoGain implements PartitionScorer {

        @Override
        public double findScore(final Instances parent, final List<Instances> children) {
            return ScoreUtils.infoGain(parent, children);
        }
    }

    class GiniEntropy implements PartitionScorer {

        @Override
        public double findScore(final Instances parent, final List<Instances> children) {
            return ScoreUtils.giniScore(children);
        }
    }

    class InfoEntropy implements PartitionScorer {

        @Override
        public double findScore(final Instances parent, final List<Instances> children) {
            return ScoreUtils.infoScore(children);
        }
    }
}
