LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
-- entity
ENTITY my_fsm IS
    PORT (
        DATA_EN : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
        CLK, BOUD_OUT, VALID : IN STD_LOGIC;
        UARTtx, BUSY
        : OUT STD_LOGIC);
END my_fsm;
ARCHITECTURE fsm OF my_fsm IS
    TYPE state_type IS (IDLE, START_S, START, RD0, RD1, RD2, RD3, RD4, RD5, RD6, RD7, STOP);
    SIGNAL state : state_type;
BEGIN
    sync_proc : PROCESS (CLK)
    BEGIN
        IF (rising_edge(CLK)) THEN

            CASE state IS
                WHEN IDLE => 
                    BUSY
                    <= '0';
                    UARTtx 
                    <= '1';
                    IF VALID = '1' THEN
                        state <= START_S;
                    END IF;
                WHEN START_S => 
                    BUSY
                    <= '1';
                    IF BOUD_OUT='1' THEN
                        state <= START;
                    END IF;
                WHEN START => 
                    BUSY
                    <= '1';
                    UARTtx <= '0';
                    IF BOUD_OUT='1' THEN
                        state <= RD0;
                    END IF;
                WHEN RD0 =>
                    BUSY
                    <= '1'; 
                    UARTtx <= DATA_EN(0);
                    IF BOUD_OUT='1' THEN
                        state <= RD1;
                    END IF;
                WHEN RD1 =>
                    BUSY
                    <= '1'; 
                    UARTtx <= DATA_EN(1);
                    IF BOUD_OUT='1' THEN
                        state <= RD2;
                    END IF;
                WHEN RD2 =>
                    BUSY
                    <= '1';
                    UARTtx <= DATA_EN(2);
                    IF BOUD_OUT='1' THEN
                        state <= RD3;
                    END IF;
                WHEN RD3 =>
                    BUSY
                    <= '1'; 
                    UARTtx <= DATA_EN(3);
                    IF BOUD_OUT='1' THEN
                        state <= RD4;
                    END IF;
                WHEN RD4 =>
                    BUSY
                    <= '1';
                    UARTtx <= DATA_EN(4);
                    IF BOUD_OUT='1' THEN
                        state <= RD5;
                    END IF;
                WHEN RD5 =>
                    BUSY
                    <= '1'; 
                    UARTtx <= DATA_EN(5);
                    IF BOUD_OUT='1' THEN
                        state <= RD6;
                    END IF;
                WHEN RD6 =>
                    BUSY
                    <= '1';
                    UARTtx <= DATA_EN(6);
                    IF BOUD_OUT='1' THEN
                        state <= RD7;
                    END IF;
                WHEN RD7 =>
                    BUSY
                    <= '1'; 
                    UARTtx <= DATA_EN(7);
                    IF BOUD_OUT='1' THEN
                        state <= STOP;
                    END IF;
                WHEN STOP => 
                    UARTtx 
                    <= '0';
                    BUSY
                    <= '0';
                    IF BOUD_OUT='1' THEN
                        state <= IDLE;
                    END IF;
                WHEN OTHERS =>
                    -- the catch-all condition
                    BUSY
                    <= '0';
                    -- arbitrary; it should never
                    state <= IDLE;
                    -- make it to these two statements
            END CASE;
        END IF;
    END PROCESS sync_proc;
END fsm;
