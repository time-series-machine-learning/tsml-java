package timeseriesweka.measures.taa;

import java.util.Arrays;
import java.util.Map;
import java.util.Stack;

// this computes the time series equivalent of NW score and also a posthoc timing penalty for the alignment
public class NW_timing {
	public static final char VERTICAL = 'v';
	public static final char DIAGONAL = 'd';
	public static final char HORIZONTAL = 'h';
	public static final double GAP_PENALTY = 0.5;
	
	public double[] score(double[] seqA, double[] seqB, double gapPenalty, int k, double UB) {
		// TODO Auto-generated method stub
		// Get sequence lengths
		int m = seqA.length;
		int n = seqB.length;

		//default scores
		double diagScore = 0;
		double horizScore = 0;
		double vertScore = 0;
		double alignmentScore = 0;

		double[][] M = new double[m+1][n+1]; //Score matrix
		char[][] D = new char[m+1][n+1];//Directions matrix
		
		//Initialize M matrix
		for (int i = 0 ; i <= m ; i++){
			M[i][0] = Integer.MAX_VALUE; //-i*gapPenalty;
			D[i][0] = VERTICAL;
		}
		for (int j = 0 ; j <= n ; j++){
			M[0][j] = Integer.MAX_VALUE; //-j*gapPenalty;
			D[0][j] = HORIZONTAL;
		}
		//for (int i = 0 ; i <= m ; i++){
		//	for (int j = 0 ; j <= n ; j++){
		//		M[i][j] = Integer.MAX_VALUE;//-j*gapPenalty;
		//	}
		//}
		
		M[0][0] = 0;
		
		//Initialize the periphery of the warping window
		for (int i = 1 ; i <= m ; i++) {
			M[i][Math.min(i+k+1,n)] = Integer.MAX_VALUE; // right edge of the window
			if((i-k-1)>1){
				M[i][i-k-1] = Integer.MAX_VALUE;
			}
		}

		int sc = 1;
		int ec = 1;
		int beg;
		int end;
		boolean smaller_found = false;
		int ec_next;
		boolean test_flag;
		// Run the algorithm
		// Go through each cell of the M matrix and use Temporal Needleman-Waterman recursion formulas to fill it in; populate the D matrix at each step!
		for (int i = 1 ; i <= m ; i++) {
			// set the start and end of the warping window for row i
			beg = Math.max(sc, i-k);
			end = Math.min(i+k,n);
			M[i][beg-1] = Integer.MAX_VALUE; // You must initialize the first value prior to the window to inifinity
			if(beg==end) {
				break;
			}
			smaller_found = false; //for each row, initially you have not found a smaller value than UB
			ec_next = i;
			test_flag = true;
			for (int j = beg; j <= end ; j++) {				
				
				diagScore = M[i-1][j-1] + Math.abs(seqA[i-1] - seqB[j-1]);
				horizScore = M[i][j-1] + gapPenalty;
				vertScore = M[i-1][j] + gapPenalty;
				
				//Set M[i][j] to the max of these scores
				if (diagScore<=horizScore && diagScore<=vertScore) {
					D[i][j] = DIAGONAL; //diagonal is the best
					M[i][j] = diagScore;
				}
				else if (vertScore<=horizScore && vertScore<=diagScore) {
					M[i][j] = vertScore;
					D[i][j] = VERTICAL; //vertical is best
				}
				else {
					D[i][j] = HORIZONTAL; //horizontal is best
					M[i][j] = horizScore;
				}
				
				//prune now
				if(!smaller_found && M[i][j]>(UB - gapPenalty*(Math.abs(j-i)))){
					if (!smaller_found){
						sc = j+1;
						test_flag=false;
						if(i!=m) {
							M[i+1][j] = Integer.MAX_VALUE; // set intial value of the cell on the next row 
						}
					}
					//if (j>=ec) {
					//	break;
					//}
				}
				else {
					smaller_found = true;
					ec_next = j+1;
				}
			}
			ec = ec_next;
		}
		
		double[] result = new double[2]; //store number of gaps
		result[0] = M[m][n]; //alignment score
		
		//Backtrack the optimal alignment and tabulate number of gaps of every length
		int i = m, j = n;
		int vGaps = 0;
		int hGaps = 0;
		
		while(i != 0 && j != 0){
			if(D[i][j] == DIAGONAL){ // diagonal 
				
				if(vGaps!=0)
					result[1] = result[1] + Math.log(Math.abs(vGaps+1)); // total possible timing penalties
				if(hGaps!=0)
					result[1]  = result[1] + Math.log(Math.abs(hGaps+1));
				
				//gaps[vGaps] = gaps[vGaps] + 1; // count vertical gaps of length vGaps
				//gaps[hGaps] = gaps[hGaps] + 1; // count horizontal gaps of length hGaps
				//reset the gap length
				vGaps = 0;
				hGaps = 0;
				i--; 
				j--;
			}else if(D[i][j] == VERTICAL){ // vertical
				vGaps = vGaps + 1;
				i--;
			}else{ // horizontal
				hGaps = hGaps + 1;
				j--;
			}
		}
				
		return result;

	}
	
	private static void printScoreMatrix(double[][] matrix){
		for(int i = 0 ; i < matrix.length ; i++){
			for(int j = 0 ; j < matrix[0].length ; j++){
				//System.out.print("" + matrix[i][j] + "  ");
				System.out.print("");
				if(matrix[i][j]>=0){
					System.out.print("+");
				}

				System.out.printf("%.8f",matrix[i][j]);
				System.out.print("  ");
			}
			System.out.println();
		}
	}
	private String getString(Stack<String> stack){
		StringBuilder seqResult = new StringBuilder();
		while(!stack.isEmpty()){
			seqResult.append(stack.pop() + " ");
		}
		return seqResult.toString().trim();

	}

	private String[] stringToArraySequence(String seq, int size) {

		String[] seqArray = new String[size];
		for(int i = 0; i<size; i++){
			seqArray = seq.split(" ");

		}
		return seqArray;
	}

	private double tPenalty(int t1, int t2, double maxTemporalPenalty) {
		// TODO Auto-generated method stub

		double temporal_penalty = 0.0;
		//given time t1 and time t2, this function return
		//temporal_penalty = - maxTemporalPenalty*Math.abs(t2-t1)/Math.max(t1,t2); //the penalty is merely a percentage of the maxTemporalPenalty heuristic (% is based on the % diff between t1 and t2)
		temporal_penalty = - maxTemporalPenalty*Math.log(Math.abs(t2-t1)+1);
		//temporal_penalty = - abs(t2-t1)/100;

		if(Double.isNaN(temporal_penalty)) //in case the transition times are both 0
			temporal_penalty = 0;

		return temporal_penalty;
	}

	private int[] getIndexSequence(String[] seq, Map<String,Integer> map){
		int[] indexSeq = new int[seq.length];
		for(int i = 0 ; i < seq.length ; i++){
			indexSeq[i]= map.get(seq[i]);
		}
		return indexSeq;

	}
	private String generateGap(int length){
		char[] fill = new char[length];
		Arrays.fill(fill, '-');
		return new String(fill);
	}

	private static void addSeqEventsToScoreMatrix(String[] seq, Map<String, Integer> map) {
		// for a single sequence, this adds the events to the labels of the scoring matrix
		for(int i = 0 ; i < seq.length ; i++){
			if(!map.containsKey(seq[i])){
				map.put(seq[i],map.size());
			}
		}	
	}
}
