LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
USE ieee.numeric_std.ALL;
ENTITY main_sm IS
    PORT (
        clk, uart_rx, baud_in : IN STD_LOGIC;
        rec_out : OUT STD_LOGIC
    );
END main_sm;
ARCHITECTURE rtl OF main_sm IS
    TYPE state_type IS (IDLE, RD0, RD1, RD2, RD3, RD4, RD5, RD6, RD7, STOP);
    SIGNAL state : state_type := IDLE;
BEGIN -- architecture rtl
    main : PROCESS (clk) IS
    BEGIN -- process main
        IF (rising_edge(CLK)) THEN
            CASE state IS
                WHEN IDLE =>
                    rec_out <= '1';
                    IF baud_in = '1' THEN
                        rec_out <= uart_rx;
                        state <= RD0;
                    END IF;
                WHEN RD0 =>
                    IF baud_in = '1' THEN
                        rec_out <= uart_rx;
                        state <= RD1;
                    END IF;
                WHEN RD1 =>
                    IF baud_in = '1' THEN
                        rec_out <= uart_rx;
                        state <= RD2;
                    END IF;
                WHEN RD2 =>
                    IF baud_in = '1' THEN
                        rec_out <= uart_rx;
                        state <= RD3;
                    END IF;
                WHEN RD3 =>
                    IF baud_in = '1' THEN
                        rec_out <= uart_rx;
                        state <= RD4;
                    END IF;
                WHEN RD4 =>
                    IF baud_in = '1' THEN
                        rec_out <= uart_rx;
                        state <= RD5;
                    END IF;
                WHEN RD5 =>
                    IF baud_in = '1' THEN
                        rec_out <= uart_rx;
                        state <= RD6;
                    END IF;
                WHEN RD6 =>
                    IF baud_in = '1' THEN
                        rec_out <= uart_rx;
                        state <= RD7;
                    END IF;
                WHEN RD7 =>
                    IF baud_in = '1' THEN
                        rec_out <= uart_rx;
                        state <= STOP;
                    END IF;
                WHEN STOP =>
                    IF baud_in = '1' THEN
                        state <= IDLE;
                        rec_out <= '0';
                    END IF;
            END CASE;
        END IF;
    END PROCESS main;
END ARCHITECTURE rtl;