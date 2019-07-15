package utilities;

import timeseriesweka.classifiers.CheckpointClassifier;

public interface Copyable<A extends Copyable<A>> {

    A copy() throws Exception;

    void copyFrom(Object object) throws Exception;

}
