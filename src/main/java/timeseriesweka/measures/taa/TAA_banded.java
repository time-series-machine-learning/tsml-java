package timeseriesweka.measures.taa;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Map;
import java.util.Stack;

public class TAA_banded {

	public static final char VERTICAL = 'v';
	public static final char DIAGONAL = 'd';
	public static final char HORIZONTAL = 'h';
	public static final double t_event= 116;

	public double score(double[] seqA, int[] sA, double[] seqB, int[] sB, double gPenalty, double tPenalty, int K) {
		// the event 't' in seqA and in seqB is represented as the value 116 (which is the ascii for t

		int m = seqA.length; 
		int n = seqB.length;

		double[][] V = new double[m + 1][n + 1]; // Score matrix
		double[][] G = new double[m + 1][n + 1]; // Match matrix
		double[][] E = new double[m + 1][n + 1]; // Horizontal gap Matrix
		double[][] F = new double[m + 1][n + 1]; // Vertical gap Matrix

		LinkedList<Integer> EPointers = new LinkedList<Integer>();
		EPointers.addFirst(0); 
		ArrayList<LinkedList<Integer>> FPointers = new ArrayList<LinkedList<Integer>>();
		LinkedList<Integer> EBlockList = new LinkedList<Integer>();
		EBlockList.addFirst(n);
		ArrayList<LinkedList<Integer>> FBlockList = new ArrayList<LinkedList<Integer>>();
		for (int i = 0; i <= n; i++) {
			LinkedList<Integer> list = new LinkedList<Integer>();
			list.addFirst(m);
			FBlockList.add(list);
			LinkedList<Integer> pointerList = new LinkedList<Integer>();
			pointerList.addFirst(0);
			FPointers.add(pointerList);
		}

		// init V
		for (int i = 0; i <= m; i++) {
			// start end seqLabels, seqNoTimeMap, delimiter
			V[i][0] = Integer.MAX_VALUE;
			E[i][0] = Integer.MAX_VALUE;
		}
		// init V
		int I = 1;
		for (int j = 0; j <= n; j++) {
			V[0][j] = Integer.MAX_VALUE;
			F[0][j] = Integer.MAX_VALUE;
			if(j!=0 && j<K) {
				I=1;
			}
			else {
				I = I+1;
			}
			
			E[I][j] = Integer.MAX_VALUE;
			F[I][j] = Integer.MAX_VALUE;
			G[I][j] = Integer.MAX_VALUE;
			V[I][j] = Integer.MAX_VALUE;
		}
		
		for (int i = 0; i <= m; i++) {
				E[i][Math.max(1,i-K-1)] = Integer.MAX_VALUE;
				F[i][Math.max(1,i-K-1)] = Integer.MAX_VALUE;
				G[i][Math.max(1,i-K-1)] = Integer.MAX_VALUE;
				V[i][Math.max(1,i-K-1)] = Integer.MAX_VALUE;
		}
		
		V[0][0] = 0;
		
		for (int i = 1; i <= m; i++) {
			// reset E Pointers
			EPointers.clear();
			EPointers.addFirst(Math.max(1,i-K-1)); // initially all will point to 0 index

			//reset E Intervals
			EBlockList.clear(); 
			EBlockList.addFirst(n); // initially there is only one interval all the way to the end
			for (int j = Math.max(1,i-K) ; j <= Math.min(i+K,n) ; j++) {
				int k = EPointers.getFirst();
				int fK = FPointers.get(j).getFirst();
				E[i][j] = CandR(k, j, V, i, sB, gPenalty, tPenalty);
				F[i][j] = CandC(fK, i, V, j, sA, gPenalty, tPenalty);
				
				if ( seqA[i-1] == t_event && seqB[j-1] == t_event ) {
					G[i][j] = V[i - 1][j - 1];
				} else if ( seqA[i-1] == t_event || seqB[j-1] == t_event ) { 
					G[i][j] = Integer.MAX_VALUE;
				} else { // the events are a match
					G[i][j] = V[i - 1][j - 1] + Math.abs(seqA[i-1] - seqB[j-1]); //match case: penalty is the norm; for timed-event sequences, you can use a scoring matrix
				}

				V[i][j] = Math.min(E[i][j], Math.min(F[i][j], G[i][j]));

				int jPrime = EBlockList.getFirst();
				int jPointer = EPointers.getFirst();
				if(j!=n && CandR(jPointer, j+1, V, i, sB, gPenalty, tPenalty) > CandR(j, j+1, V, i, sB, gPenalty, tPenalty)) { // if the candidate from j wins
					
					// j's candidate wins
					while(!EBlockList.isEmpty() && CandR(jPointer, jPrime, V, i, sB, gPenalty, tPenalty) > CandR(j, jPrime, V, i, sB, gPenalty, tPenalty)){ // if j keeps winning
						EBlockList.removeFirst();
						EPointers.removeFirst();
						if(!EBlockList.isEmpty()){
							jPrime = EBlockList.getFirst();
						}
					}
					
					if(EBlockList.isEmpty()){ // if the candidate from j is the best to the end
						EBlockList.addFirst(n); // you have one contiguous block from j through to n
					} else{ // figure out where the candidate from j stops being the best

						int BsRow = EPointers.getFirst();
						//Analytically figure out when the candidate from j will stop winning
						// for logarithmic functions, you can compute when one of them will overtake the other
						double eC = Math.exp(V[i][j]-V[i][BsRow])/tPenalty;
						
						int d = (int) Math.ceil((BsRow - j)/(1-eC)) - 1; //d is the offset of cells at which j's candidate is no longer better
						
						if (seqB[j-1]==t_event) {
							d = d*2; // you have to account for the static events in between
						} else {
							d = d*2-1; // you have to account for the static events in between
						}
						// Time series format: A t1 A t A t. 
						if(d>0) { 
							int p = j + d; // p is the cell at which the candidate from j is no longer better
							
							if(p<=n) {
								EBlockList.addFirst(p);
							}
						}
					}
					EPointers.addFirst(j);
				}

				int iPrime = FBlockList.get(j).getFirst();
				int iPointer = FPointers.get(j).getFirst();
				if(i!=m && CandC(iPointer,i+1,V,j,sA, gPenalty, tPenalty) > CandC(i,i+1,V,j,sA, gPenalty, tPenalty)){
					while(!FBlockList.get(j).isEmpty() && CandC(iPointer,iPrime, V, j, sA, gPenalty, tPenalty) > CandC(i,iPrime, V, j, sA, gPenalty, tPenalty)){
						int removedItem = FBlockList.get(j).removeFirst();
						int removedPointer = FPointers.get(j).removeFirst();
						if(!FBlockList.get(j).isEmpty()){
							iPrime = FBlockList.get(j).getFirst();
						}
					}
					if(FBlockList.get(j).isEmpty()){
						FBlockList.get(j).addFirst(m);
					}else{
						int BsCol = FPointers.get(j).getFirst();

						//Find the point at which j's candidate overtakes the candidate from B_s
						// for logarithmic functions, you can compute when one of them will overtake the other
						double eC = Math.exp(V[i][j]-V[BsCol][j])/tPenalty;
						int d = (int) Math.ceil((BsCol - i)/(1-eC)) - 1; //d is the offset of cells after which j's candidate is no longer better
						if (seqB[j-1]==t_event) {
							d = d*2; // you have to account for the static events in between
						} else {
							d = d*2-1; // you have to account for the static events in between
						}
						// d is the offset of number of cells after which j's candidate stops winning
						if(d>0) { 
							int p = j + d; // cell p is the cell where j's candidate stops winning 
							if(p<=n) {
								FBlockList.get(j).addFirst(p); // 
							}
						}
					}
					FPointers.get(j).addFirst(i);
				}
			}
		}

		return V[m][n];
	}
	/**
	 * 
	 * @param k cell that sends candidates
	 * @param j this is destination cell
	 * @param V Reference for the V matrix
	 * @param row this is row number.
	 * @return 
	 */
	private double CandR(int k, int j, double[][] V, int row, int[] s, double gPenalty, double tPenalty) {
		return V[row][k] + W(k,j,Math.abs(s[k]-s[j]), gPenalty, tPenalty);
	}
	/**
	 * 
	 * @param k row that sends candidates
	 * @param i row that receives candidates
	 * @param V reference to V matrix
	 * @param col column number
	 * @return
	 */
	private double CandC(int k, int i, double[][] V, int col, int[] s, double gPenalty, double tPenalty) {
		return V[k][col] + W(k,i,Math.abs(s[k]-s[i]), gPenalty, tPenalty);
	}

	// W is the penalty function and requires the nmber of 
	public double W(int k, int i, int Nstatic, double gPenalty, double tPenalty){

		//s is the number of static events. compute penalty
		double penalty = tPenalty*Math.log(Math.abs(i - k) - Nstatic + 1) + Nstatic*gPenalty;

		if(i==k){ // if the alignment is from the current cell to the current cell, then there is no penalty
			penalty = 0;
		}

		return penalty;
	}
	private String getString(Stack<String> stack){
		StringBuilder seqResult = new StringBuilder();
		while(!stack.isEmpty()){
			seqResult.append(stack.pop() + " ");
		}
		return seqResult.toString().trim();

	}
	private String generateGap(int length){
		char[] fill = new char[length];
		Arrays.fill(fill, '-');
		return new String(fill);
	}
	private static String[] generateTime(int length){
		String[] f = new String[length];
		Arrays.fill(f, "t");
		return f;
	}
	private int[] getIndexSequence(String[] seq, Map<String,Integer> map){
		int[] indexSeq = new int[seq.length];
		for(int i = 0 ; i < seq.length ; i++){
			indexSeq[i]= map.get(seq[i]);
		}
		return indexSeq;

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
	private static void addSeqEventsToScoreMatrix(String[] seq, Map<String, Integer> map) {
		// for a single sequence, this adds the events to the labels of the scoring matrix
		for(int i = 0 ; i < seq.length ; i++){
			if(!map.containsKey(seq[i])){
				map.put(seq[i],map.size());
			}
		}   
	}
	private static String[] seqRemoveTiming(String seq, Map<String,Integer> map){
		//remove timing values from the string if there are timing values and add the events to the labels for the score matrix
		String[] chunks = seq.split(" ");
		String[] sbSeq = new String[chunks.length];
		for(int i = 0 ; i < chunks.length ; i++){
			String chunk = chunks[i];
			String[] planAndTime = chunk.split("\\.");
			sbSeq[i] = planAndTime[0];

		}
		return sbSeq;
	}
	private static String[] seqAddTimings(String seq, int[] staticEventCount, int seqL){
		//remove timing values from the string if there are timing values and add the events to the labels for the score matrix
		String[] chunks = seq.split(" ");
		String event;
		int time; 
		String[] newSeq = new String[seqL]; // because we don't know final size
		int j;
		int staticCount = 0; // raw index in the array
		int counter = 0;
		staticEventCount[0] = 0; // the first row and column of matrix should get a static event count of 0
		for(int i = 0 ; i < chunks.length ; i++){
			String[] planAndTime = chunks[i].split("\\.");
			event = planAndTime[0];
			time = Integer.parseInt(planAndTime[1]); // amount of time
			newSeq[counter] = event; // add the event  
			counter = counter+1;
			staticCount = staticCount+1; //Number of static events
			staticEventCount[counter] = staticCount;

			for (j = 0; j < time; j++) { //j is the time number; 
				newSeq[counter] = "t";
				counter = counter+1;
				staticEventCount[counter] = staticCount;
			}
		}
		return newSeq;
	}
	private static ArrayList<String> seqIncludeTimingEvents(String seq, Map<Integer, int[]> tMap, Map<Integer, Integer> eMap){
		//remove timing values from the string if there are timing values and add the events to the labels for the score matrix
		String[] chunks = seq.split(" ");
		String[] sbSeq = new String[chunks.length];
		String event;
		int time; 
		ArrayList<String> newSeq = new ArrayList<String>(); // because we don't know final size
		int tCounts = 1;
		int j;
		int rawIndex = 0; // raw index in the array
		for(int i = 0 ; i < chunks.length ; i++){
			String chunk = chunks[i];
			String[] planAndTime = chunk.split("\\.");
			event = planAndTime[0];
			newSeq.add(event); // add the event
			rawIndex = rawIndex+1; //Raw Array Index
			eMap.put(rawIndex, tCounts); //for each static event, store the index of the upcoming timing event so it can be used for calculating timing penalties... 
			time = Integer.parseInt(planAndTime[1]); // amount of time
			for (j = tCounts; j < tCounts+time; j++) { //j is the time number; 
				rawIndex = rawIndex + 1; //count each timing event
				//map each timing event to a key in the array sequence
				tMap.put(j,new int[]{i+j, i+1}); // index of timing event in sequence, second argument is number of static events so far.
				newSeq.add(Integer.toString(j));
			}
			tCounts = tCounts+time;
		}
		System.out.println("rawIndex = " + Integer.toString(rawIndex));
		return newSeq;
	}

}
