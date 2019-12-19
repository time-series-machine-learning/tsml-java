package machine_learning.classifiers.tuned.progressive;

import tsml.classifiers.EnhancedAbstractClassifier;
import tsml.classifiers.ProgressiveBuildClassifier;
import weka.core.Instances;

public class BenchmarkingClassifier extends EnhancedAbstractClassifier implements ProgressiveBuildClassifier {

    @Override
    public boolean hasNextBuildTick() throws Exception {
        return false;
    }

    @Override
    public void nextBuildTick() throws Exception {

    }

    @Override
    public void finishBuild() throws Exception {

    }

    @Override
    public void startBuild(Instances data) throws Exception {

    }
}
