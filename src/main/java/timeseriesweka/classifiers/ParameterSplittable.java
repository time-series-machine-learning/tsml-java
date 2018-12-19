package timeseriesweka.classifiers;

import java.io.Serializable;

/**
 *
 * @author ajb
 */
public interface ParameterSplittable extends Serializable{
    public void setParamSearch(boolean b);
/* The actual parameter values should be set internally. This integer
  is just a key to maintain different parameter sets. The range starts at 1
    */
    public void setParametersFromIndex(int x);
    public String getParas();
    double getAcc();    
}
