package machine_learning.classifiers.tuned;

import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.ProgressiveBuildClassifier;
import weka.core.Instances;

public abstract class BenchmarkingClassifier extends EnhancedAbstractClassifier implements ProgressiveBuildClassifier {

    public BenchmarkingClassifier() {
        super(true);
    }

    @Override public boolean hasNextBuildTick() throws Exception {
        return false;
    }

    @Override public void nextBuildTick() throws Exception {

    }

    @Override public void finishBuild() throws Exception {

    }

    @Override public void startBuild(final Instances data) throws Exception {

    }
}
