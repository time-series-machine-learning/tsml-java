package tsml.classifiers;

import weka.core.Instance;

public interface Interperetable {

    //output summary of how the last classifier prediction was made
    String outputInterpretebilitySummary();

    //output summary of how a prediction was made for a given instance
    //most likely behaviour calls classifyInstance followed by the above method
    String outputInterpretebilitySummary(Instance inst);
}
