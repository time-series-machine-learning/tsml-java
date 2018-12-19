package statistics.tests;
	public class DataPoint implements Comparable{
		public double d;	
		public int level;	//i index: sample indicator
		public int position;//j index: data position indicator
		public double rank;
		public DataPoint(double data, int treatLevel, int pos)
		{
			d=data;
			level=treatLevel;
			position=pos;
			
		}	
		public int sampleNumber(){ return level;}
		public int level(){ return level;}
		
		public int compareTo(Object other){
			if(this.d<((DataPoint)other).d)
				return -1;
			else if	(this.d>((DataPoint)other).d)
				return 1;
			return 0;
		}	
		public boolean equals(Object other){
			return d== ((DataPoint)other).d;
		}	
	}		
