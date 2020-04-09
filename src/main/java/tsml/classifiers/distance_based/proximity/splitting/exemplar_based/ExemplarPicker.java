package tsml.classifiers.distance_based.proximity.splitting.exemplar_based;

import java.util.List;
import java.util.Map;
import java.util.Random;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;

/**
 * Purpose: pick one to many groups of exemplars from some given data. Each group represents a branch.
 * <p>
 * Contributors: goastler
 */
public interface ExemplarPicker {

    List<List<Instance>> pickExemplars(Instances instances);
}
