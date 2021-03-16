package tsml.classifiers.distance_based.utils.classifiers.configs;

import weka.classifiers.Classifier;

import java.io.Serializable;


public interface Configurer<A> extends Serializable {

    void configure(A obj);

}
