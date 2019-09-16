/*******************************************************************************
 * Copyright (C) 2017 Chang Wei Tan, Francois Petitjean, Matthieu Herrmann, Germain Forestier, Geoff Webb
 * 
 * This file is part of FastWWSearch.
 * 
 * FastWWSearch is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * 
 * FastWWSearch is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with FastWWSearch.  If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/
package timeseriesweka.classifiers.distance_based.FastWWS.tools;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;

/**
 * Code for the paper "Efficient search of the best warping window for Dynamic Time Warping" published in SDM18
 * 
 * Convert dataset in UCR format to CSV format
 * 
 * @author Chang Wei Tan, Francois Petitjean, Matthieu Herrmann, Germain Forestier, Geoff Webb
 *
 */
public class UCR2CSV {
	public static void run(File f, File fout) {
		BufferedReader in = null;
		PrintWriter out = null;
		String line;
		String[] temp;
		boolean firstLine = true;
		try {
			in = new BufferedReader(new FileReader(f));
			out = new PrintWriter(new FileOutputStream(fout), true);

			while ((line = in.readLine()) != null) {
				if (!line.isEmpty()) {
					if(firstLine){
						int k = 0;
						while (line.charAt(k) == ' ')
							k++;
						line = line.substring(k);
						temp = line.split(",");
						out.print("class");
						for (int j = 1; j < temp.length; j++) {
							out.print(",t"+(j-1));
						}
						out.println();
						firstLine=false;
					}
					int k = 0;
					while (line.charAt(k) == ' ')
						k++;
					line = line.substring(k);
					temp = line.split(",");
					out.print("'"+((int)Math.round(Double.valueOf(temp[0])))+"'");
					for (int j = 1; j < temp.length; j++) {
						out.print(","+temp[j] );
					}
					out.println();
					
				}
			}

		} catch (IOException e) {
			System.err.println("PB d'I/O");
			e.printStackTrace();
		} finally {
			try {
				in.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
			out.close();
		}
	}
}
