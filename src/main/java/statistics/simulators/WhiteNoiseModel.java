/*
Basic model that just adds normal noise. This is actually the default behaviour in 
model, so we can abstract model and do it here instead (at a later date)
 */
package statistics.simulators;

/**
 *
 * @author ajb
 */
public class WhiteNoiseModel extends  Model{

    public WhiteNoiseModel(){
        super();
    }

    @Override
    public void setParameters(double[] p) {//Mean and variance of the noise
        setVariance(p[0]);
        
    }
    
    
}
