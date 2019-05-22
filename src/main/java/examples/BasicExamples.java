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


package examples;

import java.net.MalformedURLException;
import java.net.URL;

/**
 * Class to demonstrate basic usage of the uea-tsc code base.   
 * @author ajb
 */
public class BasicExamples {
    public static void dataFormatAndLoading() throws MalformedURLException{
        System.out.println("All TSC problems are stored in Weka's ARFF format");
        System.out.println("This lists the attribute names and data type as meta data. The class value is always the last attribute");
        System.out.println("These can be loaded directly into Instances objects");
        System.out.println("We use Gunpoint for the univariate example ");
        URL gunpoint= new URL("http://www.timeseriesclassification.com/description.php?Dataset=GunPoint");
        
    }
    
    
}
