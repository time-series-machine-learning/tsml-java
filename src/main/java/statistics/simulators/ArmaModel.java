package statistics.simulators;

//Model ARMA series naively as Box Jenkins representation

// Note really need infinite AR representation

import fileIO.OutFile;
import statistics.distributions.NormalDistribution;

public class ArmaModel extends Model{

	double[] ar;
	double[] ma;

//Not sure how to map one set onto the other tho!
//Why include this, just use the ar parameters instead
	int p, q;
	double[] xRecord;
	double[] eRecord;
        
        public static double GLOBALVAR=1;
        public static void setGlobalVariance(double v){GLOBALVAR=v;}
        
	double sigma=GLOBALVAR;
        
	public ArmaModel(int p, int q)
	{
		this.p=p;
		this.q=q;
		if(p>0)
		{
			ar=new double[p];
			xRecord=new double[p];
		}
		if(q>0)
		{
			ma=new double[q];
			eRecord=new double[q];
		}
		t=0;
                sigma=GLOBALVAR;
		error = new NormalDistribution(0,sigma);
	}
	public ArmaModel(int p)
	{
		this.p=p;
		this.q=0;

		ar=new double[p];
		xRecord=new double[p];
		t=0;
                sigma=GLOBALVAR;
		error = new NormalDistribution(0,sigma);

	}
	public ArmaModel(double[] pi)
	{
		this(pi.length);
		setParas(pi);
	}
	public ArmaModel(double[]ar, double[]ma)
	{
		this(ar.length,ma.length);
		setParas(ar,ma);
	}

        @Override
	public void setParameters(double[]pi)
	{
		if(pi.length!=p)
		{
			p=pi.length;
			ar=new double[p];
			xRecord=new double[p];
		}
		q=0;
		for(int i=0;i<p;i++)
			this.ar[i]=pi[i];

	}
	public void setParas(double[]pi)
	{
		if(pi.length!=p)
		{
			p=pi.length;
			ar=new double[p];
			xRecord=new double[p];
		}
		q=0;
		for(int i=0;i<p;i++)
			this.ar[i]=pi[i];

	}
        
        public void setParas(double[]ar, double[]ma)
	{
		if(p>0 && ar.length!=p)
		{
			p=ar.length;
			this.ar=new double[p];
			xRecord=new double[p];
		}
		if(q>0 && ma.length!=q)
		{
			q=ma.length;
			this.ma=new double[q];
			eRecord=new double[q];
		}
		for(int i=0;i<p;i++)
			this.ar[i]=ar[i];
		for(int i=0;i<q;i++)
			this.ma[i]=ma[i];
	}
	public double[] getParas(){return ar;}

	public void setSigma(double s){
		sigma=s;
		((NormalDistribution)error).setSigma(s);
	}
	public void setInitialValues(double[]initX, double[]initE)
	{
		for(int i=0;i<p;i++)
			this.xRecord[i]=initX[i];
		for(int i=0;i<q;i++)
			this.eRecord[i]=initE[i];
	}
	public double generate(double x){
		return -1;
	}

	public	double generateError()
	{
		return error.simulate();
	}
	public void reset(){
		t=0;
		randomise();

	}
	public	void randomise()
	{
            for(int i=0;i<p;i++)
                    xRecord[i]=-2*sigma+4*sigma*error.RNG.nextDouble();
            for(int i=0;i<q;i++)
                    eRecord[i]=-2*sigma+4*sigma*error.RNG.nextDouble();

	}
	public	void resetToZero()
	{
		t=0;
		for(int i=0;i<p;i++)
			xRecord[i]=0;
		for(int i=0;i<q;i++)
			eRecord[i]=0;

	}
	public	double generate()
	{
            double x=0,e;
            int t =(int) (this.t);
            if(t<p)
            {
                    this.t++;
                    return xRecord[t];
            }
            for(int i=0;i<p;i++)
                    x+=ar[p-i-1]*xRecord[(t+i)%p];

            for(int i=0;i<q;i++)
                    x+=ma[(t+i)%q]*eRecord[(t+i)%q];
            e=error.simulate();
            x+=e;
            if(p>0)
                    xRecord[t%p]=x;
            if(q>0)
                    eRecord[t%q]=e;
            this.t++;
            return x;
	}

	public String toString()
	{
            String str="";
            return str;
    }

	public static double[] differenceData(double[] d)
	{
		double[] newD = new double[d.length-1];
		for(int i=0;i<d.length-1;i++)
			newD[i]=d[i+1]-d[i];
		return newD;
	}

	public static void simulateData(int p, int q, int t)
	{
		ArmaModel a = new ArmaModel(p,q);
		double[] ar = new double[p];
		double[] ma = new double[q];

		for(int i=0;i<p;i++)
			ar[i]=1;
		for(int i=0;i<q;i++)
			ma[i]=1;

		a.setParas(ar,ma);
		a.setSigma(GLOBALVAR);
		for(int i=0;i<p;i++)
			ar[i]=10;
		for(int i=0;i<q;i++)
			ma[i]=10;
		a.setInitialValues(ar,ma);
		System.out.println("Model = ARMA("+p+","+q+")");
		for(int i=0;i<t;i++)
			System.out.println("Data = "+a.generate());

		//			Compare to "anytime" Vas?? and Keogh == ??
		//		Clustering method Implement EM and Hierarchical
		//		Experiments
		//		1: Use Marahajs models
		//		2: if ineffective, find some models it works on
		//		3: Generate a class of random models
		//		Perform experiments, write paper and send
		//		to Journal of Classification
		//	Get other data sources
	}

        static public void main(String[] args){
            simulateDataForForecastingWithMP();
            System.exit(0);
            System.out.println("Testing Arma Models");

            System.out.println("Generating Data");
            simulateData(0,1,30);

	}
        public static void simulateDataForForecastingWithMP(){
            OutFile ex= new OutFile("C:\\Temp\\ARSeries.csv");
            double[][] p={{0.5},{-0.5}};
            ArmaModel ar1= new ArmaModel(p[0]);
            ArmaModel ar2= new ArmaModel(p[1]);
            for(int i=0;i<10000;i++)
                ex.writeLine(ar1.generate()+"");
            for(int i=0;i<10000;i++)
                ex.writeLine(ar2.generate()+"");
        }
         
}