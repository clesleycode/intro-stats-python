
import scipy
from scipy.stats import poisson
def poisson_probability(alarmlimit,no_of_calls,current_time):
       hourly_rate = float(alarmlimit)/24.0
       remain_hours = 24 - int(current_time)    
       rate_of_success = float(remain_hours * hourly_rate)
       if alarmlimit > no_of_calls and remain_hours!=0:
           """poisson variable is remaining no of calls to reach the limit"""
           poisson_variable = alarmlimit - no_of_calls
           probability_remain = float(scipy.stats.distributions.poisson.pmf(poisson_variable, rate_of_success))*100
           probability_reached = float(1-scipy.stats.distributions.poisson.cdf(poisson_variable-1, rate_of_success)) * 100
       elif alarmlimit > no_of_calls and remain_hours==0:    
            poisson_variable = alarmlimit - no_of_calls
            probability_remain = 0
            probability_reached = 0
       else:
           poisson_variable = 0
           probability_reached = 100
       return (poisson_variable,hourly_rate,remain_hours,probability_reached, rate_of_success)
if __name__=="__main__":
   alarmlimit=8
   no_of_calls=2
   current_time=2#its hours
   poisson_variable,hourly_rate,remain_hours,probability_reached,rate_of_success = poisson_probaility(alarmlimit,no_of_calls,current_time)
   print("poisson Variable x is %d"%(poisson_variable))
   print("Rate of Success in hour %f"%(hourly_rate))
   print("Rate of Success %f"%(rate_of_success))
   print("Remaining hours %d"%(remain_hours))
   print("Probability Reached %f"%(probability_reached))