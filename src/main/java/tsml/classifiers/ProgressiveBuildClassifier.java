package tsml.classifiers;

import com.sun.istack.internal.NotNull;
import weka.classifiers.Classifier;
import weka.core.Instances;

public interface ProgressiveBuildClassifier
    extends Classifier {

    default boolean hasNextBuildTick() throws Exception {
        return false;
    }

    default void nextBuildTick() throws Exception {

    }

    default void finishBuild() throws Exception {

    }

    default void startBuild(@NotNull Instances data) throws
                                                     Exception {}

    @Override
    default void buildClassifier(@NotNull Instances data) throws Exception {
        startBuild(data);
        if (hasNextBuildTick()) {
            do {
                nextBuildTick();
            }
            while (hasNextBuildTick());
            finishBuild();
        }
    }
}
