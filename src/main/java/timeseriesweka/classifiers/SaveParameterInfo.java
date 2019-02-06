package timeseriesweka.classifiers;
/**
 *
 * @author ajb
* Interface used for checkpointing a classifier. The getParameters is used in
* the Experiments class. This could be overlapping with another interface and 
* could possibly be depreciated. 
*
*/
public interface SaveParameterInfo {
    String getParameters();
    
}
