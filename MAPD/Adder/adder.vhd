LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
USE ieee.numeric_std.ALL;
ENTITY adder IS
    GENERIC (
        width : INTEGER := 8);
    PORT (
        a
        : IN STD_LOGIC_VECTOR(width - 1 DOWNTO 0);
        b
        : IN STD_LOGIC_VECTOR(width - 1 DOWNTO 0);
        sum : OUT STD_LOGIC_VECTOR(width - 1 DOWNTO 0));
END ENTITY adder;
ARCHITECTURE rtl OF adder IS
BEGIN
    -- architecture rtl
    sum <= STD_LOGIC_VECTOR(unsigned(a) + unsigned(b));
END ARCHITECTURE rtl;

