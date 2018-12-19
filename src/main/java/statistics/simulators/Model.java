package statistics.simulators;
import java.util.Random;
import statistics.distributions.*;
/***
 * @author ajb
 * 
 * Base class for Data model to generate simulated data.
 * 
 * In order to be able to recreate data, all random numbers should be generated
 * with calls to error.RNG.nextDouble() etc.
 * 
 */
abstract public class Model{
	protected double t=0;
	Distribution error;    
        public static double defaultSigma=1;
        public static void setDefaultSigma(double x){
            defaultSigma=x;
        }
        static int seed=1;
        static int count=1;
        double variance;
        public static Random rand=new Random(seed);
//        public static Random rand=new MersenneTwister();
        public Model(){
            variance =defaultSigma;
            error=new NormalDistribution(0,variance);
//Need different seeds for each model so using a bit of a hack singleton            
            if(seed>=0){
                error.setRandomSeed(count*(seed+1));
                count++;
            }
        }
        public static void setGlobalRandomSeed(int s){
            seed=s;
            rand=new Random(seed);
            count=1;
            
        }
        public static int getRandomSeed(){return seed;}
//Bt of a hack, what if non normal error?        
        public void setVariance(double x){
            variance=x;
            error = new NormalDistribution(0,variance);
        }
        public double getVariance(){ return variance;}
/*Generate a single data
//Assumes a model independent of previous observations. As
//such will not be relevant for ARMA or HMM models, which just return -1.
* Should probably remove. 
*/  
	double generate(double x){
            return error.simulate();
        }

//This will generate the next sequence after currently stored t value
	double generate(){
            return error.simulate();
            
        }

	public void reset(){ t=0;}
	public void setError(Distribution d){ error = d;}
//Generates a series of length n
	public	double[] generateSeries(int n)
	{
           double[] d = new double[n];
           for(int i=0;i<n;i++)
              d[i]=generate();
           return d;
        }
        public String getModelType(){ return "RandomNoise";}
        public String getAttributeName(){return "t";} 
        public String getHeader(){
            String header="%"+getModelType();
            header+="%"+"Error="+error.getClass()+" Sigma = "+error.getVariance()+"\n";
            return header;
        }
/**
 * Subclasses must implement this, how they take them out of the array is their business.
 * @param p 
 */        
        abstract public void setParameters(double[] p);


 }