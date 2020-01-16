package tsml.classifiers;

import utilities.Copy;
import utilities.Debugable;
import utilities.NotNull;
import utilities.params.ParamHandler;
import utilities.params.ParamSet;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.Randomizable;

public interface IncClassifier
    extends Classifier, Copy, Loggable, Debugable, Randomizable, RebuildableClassifier, ParamHandler {

    // make sure to start / stop timers / watchers at the beginning / end of each of these methods as they can be
    // called from anywhere! I.e. someone might call hasNextBuildTick(), wait 1 min, then call nextBuildTick(). You
    // need to stop timers in that time otherwise you're timings are out.

    default boolean hasNextBuildTick() throws Exception {
        return false;
    }

    default void nextBuildTick() throws Exception {

    }

    default void finishBuild() throws Exception {

    }

    default void startBuild(@NotNull Instances trainData) throws
                                                          Exception {}

    default void incBuildClassifier(Instances trainData) throws Exception {
        if(isRebuild()) {
            startBuild(trainData);
            setRebuild(false);
            if (hasNextBuildTick()) {
                do {
                    nextBuildTick();
                }
                while (hasNextBuildTick());
                finishBuild();
            }
        }
    }

    @Override
    default void buildClassifier(@NotNull Instances trainData) throws Exception {
        incBuildClassifier(trainData);
    }
}
