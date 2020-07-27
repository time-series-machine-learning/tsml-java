package distance.elastic;

public class Euclidean {
	public Euclidean() {

	}	
		
	public synchronized double distance(double[] s, double[] t, double bsf){
		int i = 0;
		double total = 0;

		//assume s.length == t.length for this implementation
		//TODO note <=, if bsf = 0, < will cause problems when early abandoning
		for (i = 0; i < s.length & total <= bsf; i++){
			total += (s[i] - t[i]) * (s[i] - t[i]);
		}
		
//		System.out.println("Euclidean: early abandon after: " + i + " from: " + s.length);

//		return Math.sqrt(total);
		return total;
	}
}
