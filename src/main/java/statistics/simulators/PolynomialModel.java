package statistics.simulators;
import statistics.distributions.*;


public class PolynomialModel extends Model{

	double[] coefficients;
	double[] powers;
//	Default min max range for coefficients and powers
    static double defaultMin=-4;
    static double defaultMax=4;
//DO NOT USE UNTIL DEBUGGED. Then make public
    private PolynomialModel(){super();}
    private PolynomialModel(double[] c, double[] p){
            super();
            coefficients=c;
            powers=p;
            t=0;
	}
    private PolynomialModel(double[] c, double[] p, Distribution e){
            super();
            coefficients=c;
		powers=p;
		error=e;
		t=0;
	}
    @Override
    public void setParameters(double[] p) {
        if(p.length%2!=0)
                throw new UnsupportedOperationException("Error, the Polynomial model input must be of even length, first half the powers, second half coefficients,");
        coefficients=new double[p.length/2];
        powers=new double[p.length/2];
        System.arraycopy(p,0,powers,0, p.length/2);
        System.arraycopy(p, p.length/2,coefficients, 0, p.length/2);
        error=new NormalDistribution(0,1);
        t=0;
    }

    @Override
	public double generate(double x)
	{
		double p;
		p=generateDeterministic(x);
		p+=error.simulate();
		t=x;
		return p;
	}
    @Override
	public double generate()
	{
		double p=generateDeterministic(t);
		p+=error.simulate();
		t++;
		return p;
	}
	
	public double generateDeterministic(double x)
	{
		double p=0;
		if(x==0)	//Slight hack, what about constants?
			return 0;
		for(int i=0;i<powers.length;i++)
			p+=coefficients[i]*Math.pow(x,powers[i]);
		return p;
	}
	public void setTime(int t){this.t=t;}
	public double[] getCoefficients(){return coefficients;}
	public double[] getPowers(){return powers;}
	
    @Override
	public String toString()
	{
		String str="";
		for(int i=0;i<powers.length;i++)
			str+=coefficients[i]+"*x^"+powers[i]+"\t+\t";
		str+=error;
		return str;
	}
	public static PolynomialModel generateRandomModel(int r)
	{
		return generateRandomModel(r,defaultMin,defaultMax,defaultMin,defaultMax);
	}
	public static PolynomialModel generateRandomModel(int r, double min, double max)
	{
		return generateRandomModel(r,min,max,min,max);
	}
	public static PolynomialModel generateRandomModel(int r, double minC, double maxC,double minP, double maxP)
	{
		double[] coeffs,powers;
		coeffs=new double[r];
		powers=new double[r];
		for(int i=0;i<r;i++)
		{
//HERE CHANGE STRUCTURE
//			coeffs[i]=minC+(maxC-minC)*Distribution.RNG.nextDouble();
//			powers[i]=minP+(maxP-minP)*Distribution.RNG.nextDouble();
		}
		return  new PolynomialModel(coeffs,powers, new NormalDistribution(0,1));
		
	}

public static void main(String[] args){
    System.out.println(" Test Harness not implemented yet");
}	 


}