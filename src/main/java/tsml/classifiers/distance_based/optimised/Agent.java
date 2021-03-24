package tsml.classifiers.distance_based.optimised;

import tsml.classifiers.distance_based.utils.system.logging.Loggable;
import tsml.classifiers.distance_based.utils.system.random.Randomised;
import tsml.data_containers.TimeSeriesInstances;
import weka.core.Randomizable;

import java.io.Serializable;
import java.util.Iterator;
import java.util.List;

public interface Agent extends Iterator<Evaluation>, Serializable, Randomised, Loggable {
    // build anything needing the train data
    void buildAgent(TimeSeriesInstances trainData);
    
    // called when an evaluation task has been completed, i.e. results have been populated in an evaluation
    void feedback(Evaluation evaluations);
    
    List<Evaluation> getBestEvaluations();
    
}
