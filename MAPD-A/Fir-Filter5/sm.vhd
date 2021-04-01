LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
-- entity
ENTITY fsm IS
    PORT (
        data_en : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
        clk, baud_out, valid : IN STD_LOGIC;
        uart_tx, busy
        : OUT STD_LOGIC);
END fsm;
ARCHITECTURE rtl OF fsm IS
    TYPE state_type IS (idle, start_s, start, rd0, rd1, rd2, rd3, rd4, rd5, rd6, rd7, stop);
    SIGNAL state : state_type;
BEGIN
    sync_proc : PROCESS (clk)
    BEGIN
        IF (rising_edge(clk)) THEN

            CASE state IS
                WHEN idle =>
                    busy
                    <= '0';
                    uart_tx
                    <= '1';
                    IF valid = '1' THEN
                        state <= start_s;
                    END IF;
                WHEN start_s =>
                    busy
                    <= '1';
                    IF baud_out = '1' THEN
                        state <= start;
                    END IF;
                WHEN start =>
                    busy
                    <= '1';
                    uart_tx <= '0';
                    IF baud_out = '1' THEN
                        state <= rd0;
                    END IF;
                WHEN rd0 =>
                    busy
                    <= '1';
                    uart_tx <= data_en(0);
                    IF baud_out = '1' THEN
                        state <= rd1;
                    END IF;
                WHEN rd1 =>
                    busy
                    <= '1';
                    uart_tx <= data_en(1);
                    IF baud_out = '1' THEN
                        state <= rd2;
                    END IF;
                WHEN rd2 =>
                    busy
                    <= '1';
                    uart_tx <= data_en(2);
                    IF baud_out = '1' THEN
                        state <= rd3;
                    END IF;
                WHEN rd3 =>
                    busy
                    <= '1';
                    uart_tx <= data_en(3);
                    IF baud_out = '1' THEN
                        state <= rd4;
                    END IF;
                WHEN rd4 =>
                    busy
                    <= '1';
                    uart_tx <= data_en(4);
                    IF baud_out = '1' THEN
                        state <= rd5;
                    END IF;
                WHEN rd5 =>
                    busy
                    <= '1';
                    uart_tx <= data_en(5);
                    IF baud_out = '1' THEN
                        state <= rd6;
                    END IF;
                WHEN rd6 =>
                    busy
                    <= '1';
                    uart_tx <= data_en(6);
                    IF baud_out = '1' THEN
                        state <= rd7;
                    END IF;
                WHEN rd7 =>
                    busy
                    <= '1';
                    uart_tx <= data_en(7);
                    IF baud_out = '1' THEN
                        state <= stop;
                    END IF;
                WHEN stop =>
                    uart_tx
                    <= '0';
                    busy
                    <= '1';
                    IF baud_out = '1' THEN
                        state <= idle;
                    END IF;
                WHEN OTHERS =>
                    -- the catch-all condition
                    busy
                    <= '0';
                    -- arbitrary; it should never
                    state <= idle;
                    -- make it to these two statements
            END CASE;
        END IF;
    END PROCESS sync_proc;
END rtl;