package distance.elastic;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import core.AppContext;
import core.contracts.Dataset;
 
public class DistanceMeasure {
	
	public final MEASURE distance_measure;

	private transient Euclidean euc;
	private transient DTW dtw;
	private transient DTW dtwcv;	
	private transient DDTW ddtw;
	private transient DDTW ddtwcv;	
	private transient WDTW wdtw;
	private transient WDDTW wddtw;
	private transient LCSS lcss;
	private transient MSM msm;
	private transient ERP erp;
	private transient TWE twe;
	
	public int windowSizeDTW =-1,
			windowSizeDDTW=-1, 
			windowSizeLCSS=-1,
			windowSizeERP=-1;
	public double epsilonLCSS = -1.0,
			gERP=-1.0,
			nuTWE,
			lambdaTWE,
			cMSM,
			weightWDTW,
			weightWDDTW;

	public DistanceMeasure (MEASURE m) throws Exception{
		this.distance_measure = m;
		initialize(m);
	}
	
	public void initialize (MEASURE m) throws Exception{
		switch (m) {
			case euclidean:
			case shifazEUCLIDEAN:
				euc = new Euclidean();
				break;
			case erp:
			case shifazERP:
				erp = new ERP();
				break;
			case lcss:
			case shifazLCSS:
				lcss = new LCSS();
				break;
			case msm:
			case shifazMSM:
				msm = new MSM();
				break;
			case twe:
			case shifazTWE:
				twe = new TWE();
				break;
			case wdtw:
			case shifazWDTW:
				wdtw = new WDTW();
				break;
			case wddtw:
			case shifazWDDTW:
				wddtw = new WDDTW();
				break;
			case dtw:
			case shifazDTW:
				dtw = new DTW();
				break;
			case dtwcv:
			case shifazDTWCV:
				dtwcv = new DTW();
				break;
			case ddtw:
			case shifazDDTW:
				ddtw  = new DDTW();
				break;
			case ddtwcv:
			case shifazDDTWCV:
				ddtwcv = new DDTW();
				break;
			default:
				throw new Exception("Unknown distance measure");
//				break;
		}
		
	}
	public void select_random_params(Dataset d, Random r) {
		switch (this.distance_measure) {
		case euclidean:
		case shifazEUCLIDEAN:

			break;
		case erp:
		case shifazERP:
			this.gERP = erp.get_random_g(d, r);
			this.windowSizeERP =  erp.get_random_window(d, r);
			break;
		case lcss:
		case shifazLCSS:
			this.epsilonLCSS = lcss.get_random_epsilon(d, r);
			this.windowSizeLCSS = lcss.get_random_window(d, r);
			break;
		case msm:
		case shifazMSM:
			this.cMSM = msm.get_random_cost(d, r);
			break;
		case twe:
		case shifazTWE:
			this.lambdaTWE = twe.get_random_lambda(d, r);
			this.nuTWE = twe.get_random_nu(d, r);
			break;
		case wdtw:
		case shifazWDTW:
			this.weightWDTW = wdtw.get_random_g(d, r);
			break;
		case wddtw:
		case shifazWDDTW:
			this.weightWDDTW = wddtw.get_random_g(d, r);
			break;
		case dtw:
		case shifazDTW:
			this.windowSizeDTW = d.length();	
			break;
		case dtwcv:
		case shifazDTWCV:
			this.windowSizeDTW = dtwcv.get_random_window(d, r);
			break;
		case ddtw:
		case shifazDDTW:
			this.windowSizeDDTW = d.length();	
			break;
		case ddtwcv:
		case shifazDDTWCV:
			this.windowSizeDDTW = ddtwcv.get_random_window(d, r);
			break;
		default:
//			throw new Exception("Unknown distance measure");
//			break;
		}
	}

	public double distance(double[] s, double[] t){
		return this.distance(s, t, Double.POSITIVE_INFINITY);
	}
	
	public double distance(double[] s, double[] t, double bsf){
		double distance = Double.POSITIVE_INFINITY;
		
		switch (this.distance_measure) {
		case euclidean:
		case shifazEUCLIDEAN:
			distance = euc.distance(s, t, bsf);
			break;
		case erp:
		case shifazERP:
			distance = 	erp.distance(s, t, bsf, this.windowSizeERP, this.gERP);
			break;
		case lcss:
		case shifazLCSS:
			distance = lcss.distance(s, t, bsf, this.windowSizeLCSS, this.epsilonLCSS);
			break;
		case msm:
		case shifazMSM:
			distance = msm.distance(s, t, bsf, this.cMSM);
			break;
		case twe:
		case shifazTWE:
			distance = twe.distance(s, t, bsf, this.nuTWE, this.lambdaTWE);
			break;
		case wdtw:
		case shifazWDTW:
			distance = wdtw.distance(s, t, bsf, this.weightWDTW);
			break;
		case wddtw:
		case shifazWDDTW:
			distance = wddtw.distance(s, t, bsf, this.weightWDDTW);
			break;
		case dtw:
		case shifazDTW:
			distance = dtw.distance(s, t, bsf, s.length);
			break;
		case dtwcv:
		case shifazDTWCV:
			distance = 	dtwcv.distance(s, t, bsf, this.windowSizeDTW);
			break;
		case ddtw:
		case shifazDDTW:
			distance = ddtw.distance(s, t, bsf, s.length);
			break;
		case ddtwcv:
		case shifazDDTWCV:
			distance = ddtwcv.distance(s, t, bsf, this.windowSizeDDTW);
			break;
		default:
//			throw new Exception("Unknown distance measure");
//			break;
		}
		if (distance == Double.POSITIVE_INFINITY) {
			System.out.println("error ***********");
		}
		
		return distance;
	}
	
//	public double distance(int q, int c, double bsf, DMResult result){
////		return dm.distance(s, t, bsf, result);
//		return 0.0;
//	}	
	
	public String toString() {
		return this.distance_measure.toString(); //+ " [" + dm.toString() + "]";
	}
	
	//setters and getters
	
//	public void set_param(String key, Object val) {
//		this.dm.set_param(key, val);
//	}
//	
//	public Object get_param(String key) {
//		return this.dm.get_param(key);
//	}
	
	public void setWindowSizeDTW(int w){
		this.windowSizeDTW = w;
	}
	
	public void setWindowSizeDDTW(int w){
		this.windowSizeDDTW = w;
	}
	
	public void setWindowSizeLCSS(int w){
		this.windowSizeLCSS = w;
	}
	
	public void setWindowSizeERP(int w){
		this.windowSizeERP = w;
	}
	
	public void setEpsilonLCSS(double epsilon){
		this.epsilonLCSS = epsilon;
	}
	
	public void setGvalERP(double g){
		this.gERP= g;
	}
	
	public void setNuTWE(double nuTWE){
		this.nuTWE = nuTWE;
	}
	public void setLambdaTWE(double lambdaTWE){
		this.lambdaTWE = lambdaTWE;
	}
	public void setCMSM(double c){
		this.cMSM = c;
	}
	
	public void setWeigthWDTW(double g){
		this.weightWDTW = g;
	}
	
	public void setWeigthWDDTW(double g){
		this.weightWDDTW = g;
	}
	
	
	//just to reuse this data structure
	List<Integer> closest_nodes = new ArrayList<Integer>();
	
	public int find_closest_node(
			double[] query, 
			double[][] exemplars,
			boolean train) throws Exception{
		closest_nodes.clear();
		double dist = Double.POSITIVE_INFINITY;
		double bsf = Double.POSITIVE_INFINITY;		

		for (int i = 0; i < exemplars.length; i++) {
			double[] exemplar = exemplars[i];	//TODO indices must match

			if (AppContext.config_skip_distance_when_exemplar_matches_query && exemplar == query) {
				return i;
			}
							
			dist = this.distance(query, exemplar);
			
			if (dist < bsf) {
				bsf = dist;
				closest_nodes.clear();
				closest_nodes.add(i);
			}else if (dist == bsf) {
//				if (distance == min_distance) {
//					System.out.println("min distances are same " + distance + ":" + min_distance);
//				}
				bsf = dist;
				closest_nodes.add(i);
			}
		}
		
		int r = AppContext.getRand().nextInt(closest_nodes.size());
		return closest_nodes.get(r);
	}
	
	

}
