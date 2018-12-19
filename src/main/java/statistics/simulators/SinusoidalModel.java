package statistics.simulators;

import statistics.distributions.NormalDistribution;
import fileIO.*;

public class SinusoidalModel extends Model{
    boolean warpSeries=false;
    
    void setWarp(boolean w){warpSeries=w;}
 
    static double maxAmp=3;
//RANGE 0 ..1, offset gives the starting phase of the cycle. Note this can be randomised for each series at reset() by setting
// fixedOffset to false. This shifts ALL the offsets by a fixed, random amount, to retain relative effect.     
    double[] offset;
    boolean fixedOffset=true;
    public void setFixedOffset(boolean b){fixedOffset=b;}
    
//RANGE: 0 ... 1, 100*frequency gives the number of cycles between 0 and pi()
    double[] frequency;
//RANGE:0 .. 1, gives the max height: maxAmp
    double[] amplitude;
    int maxN=100;
    double interval = Math.PI/maxN;
//Default min max range
    static double defaultMin=0;
    static double defaultMax=1;

    public SinusoidalModel(double[] p){
        if(p.length%3!=0){
            System.out.println(" Error in Sinusoidal Model, the parameter list must be divisible by three (offset, frequency and amplitude");
            throw new IllegalArgumentException(" Error in Sinusoidal Model, the parameter list must be divisible by three (offset, frequency and amplitude");
        }
        offset=new double[p.length/3];
        frequency=new double[p.length/3];
        amplitude=new double[p.length/3];
        int count=0;
        for(int i=0;i<offset.length;i++)
            offset[i]=p[count++];
        for(int i=0;i<frequency.length;i++)
            frequency[i]=p[count++];
        for(int i=0;i<amplitude.length;i++)
            amplitude[i]=p[count++];
    }
    public void setInterval(int n)
    {
      maxN=n;
      interval = Math.PI/maxN;
     }
/** from the base class. Simply split the array into three    
 * 
 * @param p 
 */
    @Override
    public void setParameters(double[] p){
        if(p.length%3!=0){
            System.out.println(" Error in Sinusoidal Model, the parameter list must be divisible by three (offset, frequency and amplitude");
        }
        offset=new double[p.length/3];
        frequency=new double[p.length/3];
        amplitude=new double[p.length/3];
        int count=0;
        for(int i=0;i<offset.length;i++)
            offset[i]=p[count++];
        for(int i=0;i<frequency.length;i++)
            frequency[i]=p[count++];
        for(int i=0;i<amplitude.length;i++)
            amplitude[i]=p[count++];
    }
 	public SinusoidalModel(double[] offset, double[] frequency, double[] amplitude)
	{
            this.offset=new double[offset.length];
            System.arraycopy(offset, 0, this.offset, 0, offset.length);
            this.frequency=new double[frequency.length];
            System.arraycopy(frequency, 0, this.frequency, 0, frequency.length);
            this.amplitude=new double[amplitude.length];
            System.arraycopy(amplitude, 0, this.amplitude, 0, amplitude.length);
            t=0;
            
            error = new NormalDistribution(0,variance);
	}
	public SinusoidalModel(double[] offset, double[] frequency, double[] amplitude, double s)
	{
            this.offset=new double[offset.length];
            System.arraycopy(offset, 0, this.offset, 0, offset.length);
            this.frequency=new double[frequency.length];
            System.arraycopy(frequency, 0, this.frequency, 0, frequency.length);
            this.amplitude=new double[amplitude.length];
            System.arraycopy(amplitude, 0, this.amplitude, 0, amplitude.length);
            t=0;
            setVariance(s);
	}
	public void setSigma(double s){
		setVariance(s);
	}
    @Override
	public double generate(double x){
		double res=0;
		for(int i=0;i<offset.length;i++)
                    res+=((maxAmp)*amplitude[i])*(Math.sin(Math.PI*offset[i]+(2+100*frequency[i])*x));
		res+= error.simulate();
        return res;
	}

	public	double generateError(){	return error.simulate();}
    @Override
	public	double generate()
	{
		double res=0;
 		for(int i=0;i<offset.length;i++)
                    res+=((maxAmp)*amplitude[i])*(Math.sin(Math.PI*offset[i]+(2+100*frequency[i])*t));
		this.t+=interval;
		res+= error.simulate();
		return res;
	}
/** @Override
 * 
 * @param l
 * @return 
 */        
    @Override
	public	double[] generateSeries(int l)
	{
	    double[] data = new double[l];
            reset();
            setInterval(l);
	    for(int k=0;k<l;k++)
                data[k] =generate();
            if(!warpSeries)
                return data;
//Warping.
            double offsetPercent=10;
            int offset=(int)(data.length*offsetPercent/100);
            double[] newD=new double[data.length];
            System.arraycopy(data,0,newD,0,data.length);
//             Pick a random point somewhere between the first 10% and last 90%
            int warpPoint=(int)(offset+Math.random()*(data.length-offset*2));
// Warp offset points into offset*2 points
            newD[warpPoint+offset]=data[warpPoint];
//Warp offset points to fill the gap.            
            newD[warpPoint-offset]=data[warpPoint-offset];
            for(int i=1;i<=offset;i++){
                newD[warpPoint-offset+2*i-1]=data[warpPoint-offset+i];
                newD[warpPoint-offset+2*i]=(data[warpPoint-offset+i]+data[warpPoint-offset+i+1])/2;
 //               System.out.println(i+","+warpPoint+","+offset+","+data[warpPoint-offset+i]+","+newD[warpPoint-offset+2*i-1]+","+newD[warpPoint-offset+2*i]);
            }
            
//            OutFile of = new OutFile("");
            return newD;
	}
	static public	double[][] generateSinusoidalData(SinusoidalModel[] models,int length,int nosPerModel)
	{
	    double[][] data = new double[models.length*nosPerModel][];

        for(int i=0;i<models.length;i++)
        {

            for(int j=0;j<nosPerModel;j++)
               data[j+i*nosPerModel]= models[i].generateSeries(length);
        }
        return data;
    }
/** @Override
 * 
 */
    @Override
    public void reset(){
//If th        
       super.reset(); 
       if(!fixedOffset)  //Shift all the offsets by a random amount  
       {
           //Find the min and max of the whole offset, ta avoid shifting too much
           double min=offset[0];
           double max=offset[0];
           for(int i=1;i<offset.length;i++){
               if(min>offset[i])
                   min=offset[i];
               if(max<offset[i])
                   max=offset[i];
           }
//So now, any shift between -min to (1-max) should retain the relative offsets
           double shift=-min+(1-max+min)*Math.random();
           for(int i=1;i<offset.length;i++)
               offset[i]+=shift;
       }
    }
        
        
/** A Load of static methods 
 * 
 * @param r
 * @param min
 * @param max
 * @return 
 */        

    public static SinusoidalModel generateRandomModel(int r, double min, double max)
    {
           double[] freq=new double[r];
           double[] off = new double[r];
           double[] amp=new double[r];
           for(int i=0;i<r;i++)
           {
//              freq[i]=min+(max-min)*Distribution.RNG.nextDouble();
//              off[i]=min+(max-min)*Distribution.RNG.nextDouble();
//              amp[i]=min+(max-min)*Distribution.RNG.nextDouble();
          }
           SinusoidalModel a = new SinusoidalModel(off,freq,amp);
           return a;
    }
    public static SinusoidalModel generateRandomModelAmp(SinusoidalModel m, double maxDeviation)
    {
    	int r = m.frequency.length;
    	double[] freq=new double[r];
    	double[] off = new double[r];
    	double[] amp=new double[r];
    	double min,max;
    	for(int i=0;i<r;i++)
    	{
    		freq[i]=m.frequency[i];
    		off[i]=m.offset[i];
      		min=m.amplitude[i]-maxDeviation;
    		if(min<0) min=0;
    		max=m.amplitude[i]+maxDeviation;
    		if(max>1) max=1;    		
//    		amp[i]=min+(max-min)*Distribution.RNG.nextDouble();
    	}
    	SinusoidalModel a = new SinusoidalModel(off,freq,amp);
    	return a;
    }		
    public static SinusoidalModel generateRandomModelOff(SinusoidalModel m, double maxDeviation)
    {
    	int r = m.frequency.length;
    	double[] freq=new double[r];
    	double[] off = new double[r];
    	double[] amp=new double[r];
    	double min,max;
    	for(int i=0;i<r;i++)
    	{
    		freq[i]=m.frequency[i];
    		amp[i]=m.amplitude[i];
      		min=m.offset[i]-maxDeviation;
    		if(min<0) min=0;
    		max=m.offset[i]+maxDeviation;
    		if(max>1) max=1;    		
//    		off[i]=min+(max-min)*Distribution.RNG.nextDouble();
    	}
    	SinusoidalModel a = new SinusoidalModel(off,freq,amp);
    	return a;
    }		
    public static SinusoidalModel generateRandomModelFreq(SinusoidalModel m, double maxDeviation)
    {
    	int r = m.frequency.length;
    	double[] freq=new double[r];
    	double[] off = new double[r];
    	double[] amp=new double[r];
    	double min,max;
    	for(int i=0;i<r;i++)
    	{
    		amp[i]=m.amplitude[i];
    		off[i]=m.offset[i];
      		min=m.frequency[i]-maxDeviation;
    		if(min<0) min=0;
    		max=m.frequency[i]+maxDeviation;
    		if(max>1) max=1;    		
//    		freq[i]=min+(max-min)*Distribution.RNG.nextDouble();
    	}
    	SinusoidalModel a = new SinusoidalModel(off,freq,amp);
    	return a;
    }		
    public static SinusoidalModel generateRandomModel(SinusoidalModel m, double maxDeviation)
    {
    	
    	int r = m.frequency.length;
    	
    	double[] freq=new double[r];
    	double[] off = new double[r];
    	double[] amp=new double[r];
    	double min,max;
    	for(int i=0;i<r;i++)
    	{
    		min=m.frequency[i]-maxDeviation;
    		if(min<0) min=0;
    		max=m.frequency[i]+maxDeviation;
    		if(max>1) max=1;    		
 //   		freq[i]=min+(max-min)*Distribution.RNG.nextDouble();
    		min=m.offset[i]-maxDeviation;
    		if(min<0) min=0;
    		max=m.offset[i]+maxDeviation;
    		if(max>1) max=1;    		
 //   		off[i]=min+(max-min)*Distribution.RNG.nextDouble();
    		min=m.amplitude[i]-maxDeviation;
    		if(min<0) min=0;
    		max=m.amplitude[i]+maxDeviation;
//    		if(max>1) max=1;    		
//    		amp[i]=min+(max-min)*Distribution.RNG.nextDouble();
       	}
    	SinusoidalModel a = new SinusoidalModel(off,freq,amp);
    	return a;
    }
    public static SinusoidalModel generateRandomModel(int r,
    		double minO, double maxO,double minF, double maxF,double minA, double maxA)
    {
    	double[] freq=new double[r];
    	double[] off = new double[r];
    	double[] amp=new double[r];
    	for(int i=0;i<r;i++)
    	{
//    		freq[i]=minF+(maxF-minF)*Distribution.RNG.nextDouble();
//    		off[i]=minO+(maxO-minO)*Distribution.RNG.nextDouble();
 //   		amp[i]=0.5+minA+(maxA-minA)*Distribution.RNG.nextDouble();
    		
    		//            off[i]=min+(max-min)*Distribution.RNG.nextDouble();
    		//         amp[i]=min+(max-min)*Distribution.RNG.nextDouble();
    	}
    	SinusoidalModel a = new SinusoidalModel(off,freq,amp);
    	return a;
    }
    public static SinusoidalModel generateRandomModel(int r)
    {
    	return generateRandomModel(r,defaultMin,defaultMax);
    }
    public static SinusoidalModel perturbSinusoidalModel(SinusoidalModel base, int maxPercentDev)
    {
           int r= base.frequency.length;
           double[] freq=new double[r];
           double[] off = new double[r];
           double[] amp=new double[r];
           int percentDev;
           for(int i=0;i<r;i++)
           {
 /*              percentDev = Distribution.RNG.nextInt(maxPercentDev);
              if(Distribution.RNG.nextDouble()<0.5)
                 freq[i]=base.frequency[i]*(100.0-percentDev)/100.0;
               else
                 freq[i]=base.frequency[i]*(100.0+percentDev)/100.0;

               percentDev = Distribution.RNG.nextInt(maxPercentDev);
               if(Distribution.RNG.nextDouble()<0.5)
                  off[i]=base.offset[i]*(100.0-percentDev)/100.0;
               else
                  off[i]=base.offset[i]*(100.0+percentDev)/100.0;
                percentDev = Distribution.RNG.nextInt(maxPercentDev);
               if(Distribution.RNG.nextDouble()<0.5)
                  amp[i]=base.amplitude[i]*(100.0-percentDev)/100.0;
               else
                  amp[i]=base.amplitude[i]*(100.0+percentDev)/100.0;
*/
         }
          SinusoidalModel a = new SinusoidalModel(off,freq,amp);
          return a;
    }
    public void randomiseOffset()
    {
 //          for(int i=0;i<offset.length;i++)
 //             offset[i]=Distribution.RNG.nextDouble();
     }
    
    double[] getOffset(){ return offset;}
    double[] getFrequency(){ return frequency;}
    double[] getAmplitude(){return amplitude;}
    void setAmplitude(double[] a){ amplitude=a;}
    void setOffset(double[] o){ offset=o;}

    public String toString(){
      String str="";
     for(int i=0;i<offset.length;i++)
              str+=offset[i]+",";
      str+="\n";
      for(int i=0;i<frequency.length;i++)
              str+=frequency[i]+",";
      str+="\n";
      for(int i=0;i<amplitude.length;i++)
              str+=amplitude[i]+",";
      str+="\n";
      return str;
    }
    public static void sampleData()
    {
           OutFile of= new OutFile("randomFFTData_3Waves.csv");
           int k=2;
           int n=256;
           int l=5;
           SinusoidalModel[] models= new SinusoidalModel[k];
           double[][] data = new double[k*l][n];
           for(int i=0;i<k;i++)
           {
                models[i]= generateRandomModel(3);
                for(int j=0;j<l;j++)
                  data[i*l+j]=models[i].generateSeries(n);
          }
          for(int i=0;i<n;i++)
          {
                  for(int j=0;j<k*l;j++)
                          of.writeString(data[j][i]+",");
                  of.writeString("\n");
           }
           for(int i=0;i<k;i++)
                   of.writeLine(models[i].toString());
        }
 	static public void main(String[] args){
                System.out.println(" To do: test harness for Sinusoidal models");
            }
        }


