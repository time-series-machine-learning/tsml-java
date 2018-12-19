/*
Interface that facilitates the saving of the internal state of the classifier,
including parameters that may have been set by CV or some other means
 */
package timeseriesweka.classifiers.ensembles;

/**
 *
 * @author ajb
 */
public interface SaveableEnsemble {
    void saveResults(String tr, String te);
    String getParameters();
   
}
