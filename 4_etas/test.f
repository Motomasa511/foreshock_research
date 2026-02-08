      subroutine add_numbers(a, b, result)
Cf2py intent(in) a, b
Cf2py intent(out) result
      implicit none
      double precision a, b, result
      result = a + b
      return
      end
    