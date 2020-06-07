package tsml.classifiers.distance_based.utils.classifier_mixins;

import tsml.classifiers.distance_based.utils.random.RandomSource;
import utilities.ArrayUtilities;
import utilities.Utilities;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;

public interface DefaultClassifier extends Classifier, RandomSource {

    @Override
    default double classifyInstance(Instance instance) throws Exception {
        double[] distribution = distributionForInstance(instance);
        return Utilities.argMax(distribution, getRandom());
    }

    @Override
    default Capabilities getCapabilities() {
        return null;
    }
}
