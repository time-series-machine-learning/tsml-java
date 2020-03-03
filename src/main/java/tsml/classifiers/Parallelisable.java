package tsml.classifiers;

/**
 * Purpose: whether something can be parallelised. At the moment, this is targeted at parallel jobs on the cluster.
 * Currently, file locking is used to detect / manage parallelisation, therefore this interface need only describe
 * whether parallelisation has occured and if so whether we have yielded to another process. In future this could
 * manage whether parallelisation is enabled or not, but that adds extra complexity.
 *
 * Contributors: goastler
 */
public interface Parallelisable extends Trainable {
    boolean isFinalModel();
}
