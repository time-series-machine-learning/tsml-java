/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
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
