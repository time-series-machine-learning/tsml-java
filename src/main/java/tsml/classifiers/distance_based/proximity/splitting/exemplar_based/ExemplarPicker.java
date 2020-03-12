package tsml.classifiers.distance_based.proximity.splitting.exemplar_based;

import java.util.List;
import java.util.Random;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;

/**
 * Purpose: // todo - docs - type the purpose of the code here
 * <p>
 * Contributors: goastler
 */
public abstract class ExemplarPicker {

    public abstract List<Instance> pickExemplars(Instances instances);
}
