LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
USE ieee.numeric_std.ALL;
ENTITY generator IS
    PORT (
        clk : IN STD_LOGIC;
        y : OUT STD_LOGIC
    );
END generator;
ARCHITECTURE rtl OF generator IS
signal counter : unsigned(9 downto 0):= (others => '0');
constant divisor: unsigned(9 downto 0):= to_unsigned(867,10);
BEGIN -- architecture rtl
    main : PROCESS (clk) IS
    BEGIN -- process main
        IF rising_edge(clk) THEN
            counter <= counter + 1;
            IF counter = divisor THEN
                y <= '1';
                counter <= (others => '0');
            ELSE
                y <= '0';
            END IF;
        END IF;
    END PROCESS main;
END ARCHITECTURE rtl;