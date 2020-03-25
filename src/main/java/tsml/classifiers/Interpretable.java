package tsml.classifiers;

import weka.core.Instance;

public interface Interpretable {

    //output summary of how the last classifier prediction was made
    String outputInterpretabilitySummary();
}
