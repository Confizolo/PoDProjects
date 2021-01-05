LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
USE ieee.numeric_std.ALL;
ENTITY samp_sm IS
    PORT (
        clk, uart_in, pulse_out : IN STD_LOGIC;
        enable : OUT STD_LOGIC
    );
END samp_sm;
ARCHITECTURE rtl OF samp_sm IS
    SIGNAL counter : unsigned(9 DOWNTO 0) := (OTHERS => '0');
    CONSTANT divisor : unsigned(9 DOWNTO 0) := to_unsigned(10, 10);
    TYPE state_type IS (IDLE, CNT, STOP);
    SIGNAL state : state_type := IDLE;
BEGIN -- architecture rtl
    main : PROCESS (clk) IS
    BEGIN -- process main
        IF (rising_edge(CLK)) THEN
            CASE state IS
                WHEN IDLE =>
                    IF uart_in = '0' THEN
                        enable <= '1';
                        counter <= (OTHERS => '0');
                        state <= CNT;
                    ELSE
                        enable <= '0';
                    END IF;
                WHEN CNT =>
                    IF pulse_out = '1' THEN
                        counter <= counter + 1;
                    END IF;
                    IF counter = divisor THEN
                        state <= STOP;
                        enable <= '0';
                    ELSE
                        enable <= '1';
                    END IF;
                WHEN STOP =>
                    IF uart_in = '1' THEN
                        state <= CNT;
                    END IF;
            END CASE;
        END IF;
    END PROCESS main;
END ARCHITECTURE rtl;