package tsml.classifiers;

import utilities.Copy;
import utilities.Debugable;
import utilities.params.ParamHandler;

import java.io.Serializable;

public interface EnhanceableClassifier extends SaveParameterInfo,
                                                  TrainSeedable, Retrainable,
                                                  Debugable, Loggable, Copy,
                                                  ParamHandler, Serializable {

}
